"""Streamlit demo for the deepvision CIFAR-10 classifier.

This module is the Phase 6 refonte of the legacy ``app.py`` that lived at the
repository root. The audit (sections 4.4 and 9) flagged four critical bugs in
the original; they are corrected here:

- **B1 -- Double softmax**: the previous code applied :func:`tf.nn.softmax`
  on top of model probabilities. We now call
  :meth:`~deepvision.serving.inference.InferenceEngine.predict` which returns
  the model's softmax probabilities directly, with no second pass.
- **B2 -- Destructive 32x32 resize**: preprocessing is delegated to
  :func:`~deepvision.serving.preprocess.preprocess_for_efficientnet` which
  targets the model's native ``IMG_SIZE_EFFICIENTNET`` (160) input.
- **B3 -- Missing RGB conversion**: the same preprocessor calls
  ``convert("RGB")`` before resizing, so RGBA and grayscale uploads no longer
  crash with a shape mismatch.
- **B4 -- Deprecated ``use_column_width``**: replaced by
  ``use_container_width``.

Additional features prescribed by the audit (section 4.4 -- Refonte app.py):

- Magic-byte MIME validation in addition to the file extension.
- Multi-image batch upload, capped by ``Settings.max_batch_size``.
- Grad-CAM overlay shown next to the top-1 prediction, leveraging
  :func:`~deepvision.evaluation.interpretability.grad_cam`.
- Clickable example images (when an ``assets/streamlit/examples/`` folder is
  present at the repository root).
- Model name and version surfaced in the sidebar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import streamlit as st
from PIL import Image

from deepvision import __version__
from deepvision.config import Settings, get_settings
from deepvision.constants import CLASS_NAMES_FR, IMG_SIZE_EFFICIENTNET
from deepvision.serving.inference import InferenceEngine
from deepvision.serving.preprocess import (
    ACCEPTED_MIME_TYPES,
    ImageValidationError,
    decode_image,
    preprocess_for_efficientnet,
    validate_payload_size,
)

#: Page configuration applied once at module import.
PAGE_CONFIG: Final[dict[str, Any]] = {
    "page_title": "DeepVision CIFAR-10",
    "page_icon": "👁️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

#: Magic-byte signatures used to verify the uploaded file actually matches the
#: claimed MIME type. The browser-reported extension is trivial to spoof, so
#: header-based detection is the only reliable check.
_PNG_SIG: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_JPEG_SIG: Final[bytes] = b"\xff\xd8\xff"
_RIFF_SIG: Final[bytes] = b"RIFF"
_WEBP_SIG: Final[bytes] = b"WEBP"


def detect_mime(payload: bytes) -> str | None:
    """Return the MIME type inferred from magic bytes, or ``None`` if unknown.

    Recognises the formats accepted by the inference API: PNG, JPEG, WebP.
    """
    if len(payload) < 12:
        return None
    if payload.startswith(_PNG_SIG):
        return "image/png"
    if payload.startswith(_JPEG_SIG):
        return "image/jpeg"
    if payload[:4] == _RIFF_SIG and payload[8:12] == _WEBP_SIG:
        return "image/webp"
    return None


def validate_mime(payload: bytes) -> str:
    """Return the MIME type, or raise :class:`ImageValidationError`.

    Uses :func:`detect_mime` so a renamed ``.txt`` masquerading as ``.png``
    is rejected before we hand it to Pillow.
    """
    mime = detect_mime(payload)
    if mime is None or mime not in ACCEPTED_MIME_TYPES:
        raise ImageValidationError(
            f"Unsupported file format. Accepted: {', '.join(ACCEPTED_MIME_TYPES)}."
        )
    return mime


@st.cache_resource(show_spinner=False)
def get_engine(model_path: str | None) -> InferenceEngine:
    """Load and cache an :class:`InferenceEngine` for the Streamlit session.

    Streamlit caches based on the ``model_path`` argument, so changing the
    configured path triggers a fresh load. The engine itself caches the
    underlying Keras model lazily on the first :meth:`predict` call.
    """
    settings = get_settings()
    engine = InferenceEngine(
        model_path=Path(model_path) if model_path else None,
        model_name=settings.serving_model_name,
        model_version=settings.serving_model_version,
    )
    engine.load()
    return engine


def predict_one(
    engine: InferenceEngine,
    image: Image.Image,
) -> dict[str, Any]:
    """Run the full pre-process + inference pipeline on one Pillow image.

    The returned probabilities come straight from
    :meth:`InferenceEngine.predict` -- there is **no** second softmax (B1).
    Pre-processing targets ``IMG_SIZE_EFFICIENTNET`` (160) via
    :func:`preprocess_for_efficientnet`, which also converts to RGB (B2 + B3).
    """
    batch = preprocess_for_efficientnet(image, target_size=IMG_SIZE_EFFICIENTNET)
    items, inference_time_ms = engine.predict(batch, top_k=3)
    return {
        "top_k": items,
        "inference_time_ms": inference_time_ms,
        "batch": batch,
    }


def render_gradcam_overlay(
    engine: InferenceEngine,
    batch: np.ndarray,
    target_class: int,
) -> np.ndarray:
    """Compute and overlay a Grad-CAM heatmap for the predicted class."""
    # Local import: TensorFlow is heavy and shouldn't pay its cost when the
    # user disables Grad-CAM in the sidebar.
    from deepvision.evaluation.interpretability import (
        grad_cam,
        overlay_heatmap_on_image,
    )

    image_3d = batch[0]  # (H, W, C) uint8
    heatmap = grad_cam(engine.model, image_3d, target_class=target_class)
    return overlay_heatmap_on_image(image_3d, heatmap)


def render_sidebar(settings: Settings) -> bool:
    """Render the sidebar; returns whether Grad-CAM should be displayed."""
    st.sidebar.title("À propos")
    st.sidebar.markdown(
        f"**deepvision** v{__version__}\n\nPipeline industriel CIFAR-10 -- "
        "EfficientNetB0 transfer learning."
    )

    st.sidebar.subheader("Modèle servi")
    st.sidebar.markdown(f"- **Nom** : `{settings.serving_model_name}`")
    st.sidebar.markdown(f"- **Version** : `{settings.serving_model_version}`")
    if settings.model_path is None:
        st.sidebar.warning("Aucun `model_path` configuré : poids aléatoires (démo uniquement).")
    else:
        st.sidebar.success(f"Poids : `{settings.model_path}`")

    st.sidebar.subheader("Options")
    show_gradcam = st.sidebar.checkbox(
        "Afficher Grad-CAM",
        value=True,
        help="Carte de chaleur des régions ayant le plus contribué à la prédiction.",
    )

    st.sidebar.subheader("Limites")
    max_mb = settings.max_image_bytes / (1024 * 1024)
    st.sidebar.caption(
        f"Taille max : **{max_mb:.0f} Mo** par image, "
        f"formats : {', '.join(ACCEPTED_MIME_TYPES)}, "
        f"batch max : **{settings.max_batch_size}** images."
    )
    return show_gradcam


def render_prediction(
    engine: InferenceEngine,
    image: Image.Image,
    *,
    show_gradcam: bool,
    label: str = "",
) -> None:
    """Render the prediction UI for a single image (input + result + Grad-CAM)."""
    col_input, col_result = st.columns([1, 1])

    with col_input:
        if label:
            st.markdown(f"#### `{label}`")
        # B4 fixed: ``use_container_width`` instead of the deprecated
        # ``use_column_width``.
        st.image(image, caption="Image d'entrée", use_container_width=True)

    try:
        with st.spinner("Inférence…"):
            result = predict_one(engine, image)
    except (ImageValidationError, ValueError, RuntimeError) as exc:
        st.error(f"Échec de l'inférence : {exc}")
        return

    top_k = result["top_k"]
    top_idx, _top_name_en, top_prob = top_k[0]

    with col_result:
        st.metric(
            label="Prédiction",
            value=CLASS_NAMES_FR[top_idx],
            delta=f"{top_prob * 100:.1f}% confiance",
        )
        st.caption(f"Latence inférence : {result['inference_time_ms']:.1f} ms")
        st.markdown("**Top-3**")
        for idx, name_en, prob in top_k:
            st.progress(
                int(prob * 100),
                text=f"{CLASS_NAMES_FR[idx]} ({name_en}) -- {prob * 100:.1f}%",
            )

    if show_gradcam:
        with st.expander("Grad-CAM (régions saillantes)", expanded=False):
            try:
                with st.spinner("Calcul Grad-CAM…"):
                    overlay = render_gradcam_overlay(
                        engine,
                        result["batch"],
                        target_class=top_idx,
                    )
                st.image(
                    overlay,
                    caption=f"Grad-CAM -- classe {CLASS_NAMES_FR[top_idx]}",
                    use_container_width=True,
                )
            except (ValueError, RuntimeError) as exc:
                st.warning(f"Grad-CAM indisponible : {exc}")


def find_examples_dir() -> Path | None:
    """Return the bundled ``assets/streamlit/examples/`` folder if it exists.

    The folder is optional; absence is silently ignored so the app still runs
    without sample images.
    """
    candidate = Path(__file__).resolve().parent.parent.parent / "assets" / "streamlit" / "examples"
    return candidate if candidate.is_dir() else None


def render_examples_section(
    engine: InferenceEngine,
    *,
    show_gradcam: bool,
) -> None:
    """Render a small grid of clickable example images, when available."""
    examples_dir = find_examples_dir()
    if examples_dir is None:
        st.caption(
            "_Astuce : déposez quelques `.png` ou `.jpg` dans "
            "`assets/streamlit/examples/` pour activer la galerie d'exemples._"
        )
        return

    paths = sorted(examples_dir.glob("*.png")) + sorted(examples_dir.glob("*.jpg"))
    if not paths:
        return

    st.subheader("Exemples cliquables")
    cols = st.columns(min(len(paths), 5))
    for idx, path in enumerate(paths[:10]):
        with cols[idx % len(cols)]:
            st.image(str(path), use_container_width=True, caption=path.stem)
            if st.button("Tester", key=f"example-{path.stem}"):
                st.session_state["example_path"] = str(path)

    selected = st.session_state.get("example_path")
    if selected:
        st.markdown("---")
        with open(selected, "rb") as fh:
            payload = fh.read()
        try:
            image = decode_image(payload)
        except ImageValidationError as exc:
            st.error(f"Exemple `{Path(selected).name}` invalide : {exc}")
            return
        render_prediction(
            engine,
            image,
            show_gradcam=show_gradcam,
            label=Path(selected).name,
        )


def main() -> None:  # pragma: no cover -- exercised via streamlit run / AppTest
    """Streamlit entrypoint executed by ``streamlit run``."""
    st.set_page_config(**PAGE_CONFIG)
    settings = get_settings()
    show_gradcam = render_sidebar(settings)

    st.title("👁️ DeepVision -- CIFAR-10")
    st.markdown(
        "Démo de classification d'images sur les 10 classes CIFAR-10 (avion, "
        "voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau, "
        "camion). Backbone : EfficientNetB0 (transfer learning). "
        f"Version `{settings.serving_model_version}`."
    )

    engine = get_engine(str(settings.model_path) if settings.model_path else None)

    st.markdown("---")
    uploaded = st.file_uploader(
        "Glissez une ou plusieurs images (JPEG / PNG / WebP, ≤10 Mo chacune)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded:
        if len(uploaded) > settings.max_batch_size:
            st.error(
                f"Trop d'images : {len(uploaded)} fournies, maximum {settings.max_batch_size}."
            )
            uploaded = uploaded[: settings.max_batch_size]

        for f in uploaded:
            payload = f.read()
            try:
                validate_payload_size(payload, max_bytes=settings.max_image_bytes)
                validate_mime(payload)
                image = decode_image(payload)
            except ImageValidationError as exc:
                st.error(f"`{f.name}` rejetée : {exc}")
                continue
            render_prediction(
                engine,
                image,
                show_gradcam=show_gradcam,
                label=f.name,
            )
            st.markdown("---")
    else:
        render_examples_section(engine, show_gradcam=show_gradcam)


if __name__ == "__main__":  # pragma: no cover
    main()
