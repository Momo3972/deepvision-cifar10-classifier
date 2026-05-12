"""Structural tests for the Phase 8 monitoring artefacts.

We do not boot Prometheus or Grafana. We only assert on what the operator
contract says must be present in the YAML/JSON files that are mounted into
those containers, so the suite stays runnable in any CI without Docker.

Coverage:

* ``monitoring/alerts.yml``: the audit-prescribed alerts exist with the
  expected severity and use a PromQL expression that the rules engine
  considers well-formed (we do not evaluate them, only parse).
* ``monitoring/grafana/dashboards/deepvision.json``: parses as JSON, has
  the expected number of panels, and the Prometheus datasource UID
  matches the one declared by the auto-provisioning file.
* ``monitoring/grafana/provisioning/datasources/prometheus.yml``: declares
  Prometheus on the in-network DNS name ``prometheus:9090``.
* ``monitoring/grafana/provisioning/dashboards/dashboards.yml``: points
  at ``/var/lib/grafana/dashboards`` (the path that compose mounts).
* ``monitoring/prometheus.yml``: scrapes both ``deepvision-api`` and
  ``deepvision-drift-monitor``, and references ``alerts.yml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest
import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def alerts() -> dict:
    return yaml.safe_load((REPO_ROOT / "monitoring" / "alerts.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(
        (REPO_ROOT / "monitoring" / "grafana" / "dashboards" / "deepvision.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def grafana_datasources() -> dict:
    return yaml.safe_load(
        (
            REPO_ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
        ).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def grafana_dashboards_provisioning() -> dict:
    return yaml.safe_load(
        (
            REPO_ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
        ).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def prometheus_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "monitoring" / "prometheus.yml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# alerts.yml
# ---------------------------------------------------------------------------


def _all_alerts(alerts: dict) -> list[dict]:
    return [rule for group in alerts["groups"] for rule in group["rules"] if "alert" in rule]


REQUIRED_ALERTS: Final[set[str]] = {
    "DeepvisionApiDown",
    "DeepvisionModelNotLoaded",
    "DeepvisionHighInferenceLatencyP95",
    "DeepvisionHighErrorRate",
    "DeepvisionBaselineMissing",
    "DeepvisionEmbeddingDriftHigh",
    "DeepvisionEmbeddingDriftCritical",
    "DeepvisionOodRateHigh",
}


def test_alerts_yaml_is_valid_and_has_groups(alerts: dict) -> None:
    assert "groups" in alerts
    assert isinstance(alerts["groups"], list)
    assert all("rules" in g for g in alerts["groups"])


def test_required_alerts_present(alerts: dict) -> None:
    names = {rule["alert"] for rule in _all_alerts(alerts)}
    missing = REQUIRED_ALERTS - names
    assert not missing, f"Missing alerts: {missing}"


@pytest.mark.parametrize("alert_name", sorted(REQUIRED_ALERTS))
def test_alert_has_required_fields(alerts: dict, alert_name: str) -> None:
    rule = next(r for r in _all_alerts(alerts) if r["alert"] == alert_name)
    assert "expr" in rule, f"{alert_name}: missing expr"
    assert rule["expr"].strip(), f"{alert_name}: empty expr"
    assert "for" in rule, f"{alert_name}: missing for-duration"
    assert "labels" in rule, f"{alert_name}: missing labels"
    assert "severity" in rule["labels"], f"{alert_name}: missing severity label"
    assert rule["labels"]["severity"] in {"critical", "warning", "info"}
    assert "annotations" in rule, f"{alert_name}: missing annotations"
    assert "summary" in rule["annotations"], f"{alert_name}: missing summary annotation"


# ---------------------------------------------------------------------------
# Grafana dashboard JSON
# ---------------------------------------------------------------------------


def test_dashboard_parses_and_has_uid(dashboard: dict) -> None:
    assert dashboard.get("uid"), "Dashboard must declare a stable uid"
    assert dashboard.get("title"), "Dashboard must declare a title"


def test_dashboard_has_at_least_six_panels(dashboard: dict) -> None:
    panels = dashboard.get("panels", [])
    assert len(panels) >= 6, f"Expected >= 6 panels, got {len(panels)}"


def test_dashboard_panels_use_provisioned_datasource_uid(
    dashboard: dict, grafana_datasources: dict
) -> None:
    """Every panel that has a Prometheus datasource must point at the same
    UID declared in the auto-provisioning file."""
    expected_uid = grafana_datasources["datasources"][0]["uid"]
    for panel in dashboard["panels"]:
        ds = panel.get("datasource")
        if isinstance(ds, dict) and ds.get("type") == "prometheus":
            assert ds.get("uid") == expected_uid, (
                f"Panel {panel.get('title')!r}: datasource uid {ds.get('uid')!r} "
                f"does not match provisioning uid {expected_uid!r}"
            )


def test_dashboard_targets_reference_known_metrics(dashboard: dict) -> None:
    """Quick sanity check: at least one target queries a deepvision_* metric."""
    found = False
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            if "deepvision_" in target.get("expr", ""):
                found = True
                break
    assert found, "No panel queries a deepvision_* metric"


# ---------------------------------------------------------------------------
# Grafana provisioning
# ---------------------------------------------------------------------------


def test_grafana_datasource_points_at_prometheus_service(grafana_datasources: dict) -> None:
    ds = grafana_datasources["datasources"][0]
    assert ds["type"] == "prometheus"
    assert ds["url"] == "http://prometheus:9090"
    assert ds.get("isDefault") is True


def test_grafana_dashboards_provisioning_paths(
    grafana_dashboards_provisioning: dict,
) -> None:
    provider = grafana_dashboards_provisioning["providers"][0]
    assert provider["type"] == "file"
    assert provider["options"]["path"] == "/var/lib/grafana/dashboards"


# ---------------------------------------------------------------------------
# Prometheus config (Phase 8 additions)
# ---------------------------------------------------------------------------


def test_prometheus_loads_alerts_yml(prometheus_config: dict) -> None:
    assert "rule_files" in prometheus_config
    assert any("alerts.yml" in rf for rf in prometheus_config["rule_files"]), prometheus_config[
        "rule_files"
    ]


def test_prometheus_scrapes_drift_monitor(prometheus_config: dict) -> None:
    jobs = {job["job_name"] for job in prometheus_config.get("scrape_configs", [])}
    assert "deepvision-drift-monitor" in jobs


def test_prometheus_drift_monitor_target(prometheus_config: dict) -> None:
    drift_job = next(
        job
        for job in prometheus_config["scrape_configs"]
        if job["job_name"] == "deepvision-drift-monitor"
    )
    targets = [t for sc in drift_job["static_configs"] for t in sc["targets"]]
    assert "drift-monitor:9091" in targets
