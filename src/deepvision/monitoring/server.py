"""Prometheus exporter for embedding drift and OOD rate.

Phase 8 deliverable -- the *real* drift-monitor service that replaces the
sleeping placeholder declared in Phase 7's ``docker-compose.yml``.

Architecture
------------

The service is a small Python process that:

1. Loads the **same model** as the FastAPI inference service (via
   :class:`~deepvision.serving.inference.InferenceEngine`), but with a
   different entrypoint -- so the api Docker image is reused, no extra
   image to build.
2. Loads (or, for smoke testing, computes on the fly) a **baseline**
   embedding distribution captured at training time.
3. Every ``interval`` seconds, samples a fresh "current" batch (also
   synthetic in the smoke-test path; an operator can wire this to a
   reservoir-sampled production stream later), extracts its embeddings,
   computes Wasserstein drift vs the baseline, computes OOD rate via the
   energy score, and updates the Prometheus gauges.
4. Exposes ``/metrics`` on a configurable port (default ``9091``) so
   Prometheus can scrape it like any other target.

Metrics published
-----------------

==============================================  ==============================
metric                                          purpose
==============================================  ==============================
``deepvision_drift_score{...}``                 mean per-dim Wasserstein
``deepvision_drift_max{...}``                   worst-dim Wasserstein
``deepvision_drift_p95{...}``                   95th-percentile Wasserstein
``deepvision_ood_rate{...}``                    fraction of samples flagged
``deepvision_drift_polls_total{...}``           successful polling cycles
``deepvision_drift_errors_total{...}``          polling cycles that raised
``deepvision_baseline_n_samples{...}``          baseline cardinality
``deepvision_baseline_loaded{...}``             ``1`` once baseline is ready
==============================================  ==============================

All gauges/counters carry ``model_name`` and ``model_version`` labels so
the Grafana dashboard can surface multiple models side-by-side.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    start_http_server,
)

from deepvision import __version__
from deepvision.config import Settings, get_settings
from deepvision.monitoring.baseline import (
    Baseline,
    compute_baseline,
    extract_embeddings,
    synthetic_reference_images,
)
from deepvision.monitoring.drift import drift_score, summarize_drift, wasserstein_drift
from deepvision.monitoring.ood import energy_score, extract_logits, ood_rate
from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from deepvision.serving.inference import InferenceEngine

log = get_logger(__name__)

#: Default OOD threshold on the energy score. Calibrated on a 256-sample
#: synthetic baseline with random weights, this value yields ~5 % false
#: positives -- a sensible smoke-test default. Real deployments should
#: recalibrate against their own baseline distribution.
DEFAULT_OOD_THRESHOLD: Final[float] = -2.0

#: Default polling interval (seconds). Matches the Prometheus scrape
#: interval declared in ``monitoring/prometheus.yml``.
DEFAULT_POLL_INTERVAL: Final[float] = 60.0

#: Default exporter port. Wired to the docker-compose ``drift-monitor``
#: service and to the Phase 8 scrape job in ``monitoring/prometheus.yml``.
DEFAULT_EXPORTER_PORT: Final[int] = 9091

#: Default size of the synthetic batch sampled each polling cycle.
DEFAULT_BATCH_SIZE: Final[int] = 64


class DriftMonitor:
    """Periodic drift + OOD evaluator with Prometheus exposition.

    The class is intentionally small and stateful so the polling loop can
    be unit-tested by calling :meth:`poll_once` directly; the long-lived
    HTTP server is only spun up by :meth:`serve_forever`.

    Parameters
    ----------
    settings
        The application :class:`~deepvision.config.Settings` instance,
        used to resolve ``model_path`` and the model identity tags.
    baseline_path
        Optional path to a previously saved :class:`Baseline` ``.npz``.
        When ``None`` or the path is missing, a synthetic baseline is
        computed on the fly so the service stays useful for smoke tests.
    port
        TCP port exposed by the Prometheus HTTP exporter.
    interval
        Seconds between two polling cycles.
    batch_size
        Number of synthetic images sampled per polling cycle.
    ood_threshold
        Energy threshold above which a sample is flagged OOD.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        baseline_path: Path | None = None,
        port: int = DEFAULT_EXPORTER_PORT,
        interval: float = DEFAULT_POLL_INTERVAL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        ood_threshold: float = DEFAULT_OOD_THRESHOLD,
    ) -> None:
        self.settings = settings or get_settings()
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.port = port
        self.interval = interval
        self.batch_size = batch_size
        self.ood_threshold = ood_threshold

        self._engine: InferenceEngine | None = None
        self._baseline: Baseline | None = None
        self._registry = CollectorRegistry()

        labels = ("model_name", "model_version")

        self._drift_score_g = Gauge(
            "deepvision_drift_score",
            "Mean per-dimension Wasserstein distance vs baseline embeddings.",
            labels,
            registry=self._registry,
        )
        self._drift_max_g = Gauge(
            "deepvision_drift_max",
            "Worst single-dimension Wasserstein distance vs baseline.",
            labels,
            registry=self._registry,
        )
        self._drift_p95_g = Gauge(
            "deepvision_drift_p95",
            "95th-percentile per-dimension Wasserstein distance vs baseline.",
            labels,
            registry=self._registry,
        )
        self._ood_rate_g = Gauge(
            "deepvision_ood_rate",
            "Fraction of samples flagged out-of-distribution by the energy score.",
            labels,
            registry=self._registry,
        )
        self._poll_counter = Counter(
            "deepvision_drift_polls_total",
            "Number of polling cycles that completed successfully.",
            labels,
            registry=self._registry,
        )
        self._error_counter = Counter(
            "deepvision_drift_errors_total",
            "Number of polling cycles that raised an exception.",
            labels,
            registry=self._registry,
        )
        self._baseline_n_g = Gauge(
            "deepvision_baseline_n_samples",
            "Number of samples in the active baseline distribution.",
            labels,
            registry=self._registry,
        )
        self._baseline_loaded_g = Gauge(
            "deepvision_baseline_loaded",
            "1 once the baseline has been loaded or computed; 0 before.",
            labels,
            registry=self._registry,
        )
        self._info_g = Gauge(
            "deepvision_drift_monitor_info",
            "Static metadata about the drift monitor (always 1).",
            ("model_name", "model_version", "package_version"),
            registry=self._registry,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load the engine + baseline. Idempotent, safe to call before serve."""
        from deepvision.serving.inference import InferenceEngine

        if self._engine is None:
            log.info("Initialising InferenceEngine (model_path=%s)", self.settings.model_path)
            self._engine = InferenceEngine(
                model_path=self.settings.model_path,
                model_name=self.settings.serving_model_name,
                model_version=self.settings.serving_model_version,
            )
            self._engine.load()

        labels = (self._engine.model_name, self._engine.model_version)
        self._info_g.labels(*labels, __version__).set(1)
        self._baseline_loaded_g.labels(*labels).set(0)

        if self._baseline is None:
            self._baseline = self._load_or_compute_baseline()
            self._baseline_n_g.labels(*labels).set(self._baseline.n_samples)
            self._baseline_loaded_g.labels(*labels).set(1)

    def _model_image_size(self) -> int:
        """Spatial input size expected by the loaded model (e.g. 32 or 160).

        The DriftMonitor must feed images that match this size, otherwise
        Keras raises a shape-mismatch ``ValueError`` on the first forward
        pass. We derive the size from ``model.input_shape`` rather than
        hard-coding ``IMG_SIZE_EFFICIENTNET`` so the monitor remains valid
        if the model is rebuilt at another resolution.
        """
        assert self._engine is not None
        try:
            shape = self._engine.model.input_shape  # (None, H, W, C)
            return int(shape[1])
        except (AttributeError, TypeError, IndexError):
            # Fall back on the EfficientNet default; if it crashes the
            # synthetic-baseline path will also crash and the operator will
            # see a clear shape error in the logs.
            from deepvision.constants import IMG_SIZE_EFFICIENTNET

            return IMG_SIZE_EFFICIENTNET

    def _load_or_compute_baseline(self) -> Baseline:
        if self.baseline_path is not None and self.baseline_path.exists():
            log.info("Loading baseline from %s", self.baseline_path)
            return Baseline.load(self.baseline_path)

        log.warning(
            "No baseline file at %s -- computing a synthetic baseline (smoke test).",
            self.baseline_path,
        )
        assert self._engine is not None  # initialize() ran first
        size = self._model_image_size()
        log.info("Synthetic baseline at native model size %dx%d", size, size)
        images = synthetic_reference_images(n=256, seed=42, image_size=size)
        return compute_baseline(
            self._engine.model,
            images,
            model_name=self._engine.model_name,
            model_version=self._engine.model_version,
        )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll_once(self) -> dict[str, float]:
        """Run one drift + OOD evaluation cycle and update the gauges.

        Returns
        -------
        dict
            ``{"drift_mean": ..., "drift_max": ..., "drift_p95": ...,
            "ood_rate": ...}`` for logging / debugging.

        Raises
        ------
        RuntimeError
            If :meth:`initialize` has not been called.
        """
        if self._engine is None or self._baseline is None:
            raise RuntimeError("DriftMonitor.initialize() must be called first.")

        labels = (self._engine.model_name, self._engine.model_version)
        try:
            current_images = synthetic_reference_images(
                n=self.batch_size,
                seed=int(time.time()) & 0xFFFF,
                image_size=self._model_image_size(),
            )
            embeddings = extract_embeddings(self._engine.model, current_images)
            logits = extract_logits(self._engine.model, current_images)
            energies = energy_score(logits)

            distances = wasserstein_drift(self._baseline.embeddings, embeddings)
            summary = summarize_drift(distances)
            ood_fraction = ood_rate(energies, threshold=self.ood_threshold)

            self._drift_score_g.labels(*labels).set(summary["mean"])
            self._drift_max_g.labels(*labels).set(summary["max"])
            self._drift_p95_g.labels(*labels).set(summary["p95"])
            self._ood_rate_g.labels(*labels).set(ood_fraction)
            self._poll_counter.labels(*labels).inc()

            log.info(
                "drift mean=%.4f max=%.4f p95=%.4f ood_rate=%.2f%%",
                summary["mean"],
                summary["max"],
                summary["p95"],
                ood_fraction * 100,
            )
            return {
                "drift_mean": summary["mean"],
                "drift_max": summary["max"],
                "drift_p95": summary["p95"],
                "ood_rate": ood_fraction,
            }
        except Exception:
            self._error_counter.labels(*labels).inc()
            log.exception("Drift polling cycle failed")
            raise

    def serve_forever(self) -> None:  # pragma: no cover -- long-running loop
        """Start the Prometheus HTTP exporter and loop forever.

        Calls :meth:`initialize` if it has not been called yet.
        """
        if self._engine is None:
            self.initialize()
        log.info(
            "Starting drift-monitor exporter on :%d (interval=%ss, batch=%d)",
            self.port,
            self.interval,
            self.batch_size,
        )
        start_http_server(self.port, registry=self._registry)
        while True:
            # Already counted + logged inside poll_once on raise; suppress here
            # so a transient failure does not take the exporter down.
            with contextlib.suppress(Exception):
                self.poll_once()
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# CLI helper -- invoked by ``python -m deepvision drift-monitor``.
# ---------------------------------------------------------------------------


def run(  # pragma: no cover -- thin CLI wrapper
    *,
    port: int = DEFAULT_EXPORTER_PORT,
    interval: float = DEFAULT_POLL_INTERVAL,
    baseline_path: Path | None = None,
    ood_threshold: float = DEFAULT_OOD_THRESHOLD,
) -> None:
    """Entry point used by the ``deepvision drift-monitor`` Typer command.

    Kept separate from :class:`DriftMonitor` so unit tests can exercise the
    class without launching a long-running HTTP server.

    Convenience: a side-effect of importing this module is that
    ``drift_score`` is in scope -- silence the linter by referencing it
    once, since the public re-export documents the available helpers.
    """
    _ = drift_score  # re-exported helper, see module docstring
    monitor = DriftMonitor(
        baseline_path=baseline_path,
        port=port,
        interval=interval,
        ood_threshold=ood_threshold,
    )
    monitor.serve_forever()
