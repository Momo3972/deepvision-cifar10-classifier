"""
Diagnostic des capacités de la machine pour la refonte deepvision-cifar10-classifier.

Usage :
    # Active d'abord ton environnement virtuel :
    .venv\\Scripts\activate          (Windows PowerShell)

    # Puis lance :
    python scripts/check_machine.py

Ce script ne dépend que de la lib standard + (optionnellement) tensorflow et psutil
si déjà installés. Aucune installation forcée.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------- helpers ---------------------------------------------------------

C = {
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}
# Désactive les couleurs si la sortie n'est pas un terminal
if not sys.stdout.isatty() or (os.name == "nt" and not os.environ.get("TERM")):
    for k in C:
        C[k] = ""


def section(title: str) -> None:
    print(f"\n{C['bold']}{C['blue']}{'─' * 70}{C['reset']}")
    print(f"{C['bold']}{C['blue']} {title}{C['reset']}")
    print(f"{C['bold']}{C['blue']}{'─' * 70}{C['reset']}")


def ok(msg: str) -> None:
    print(f"  {C['green']}✓{C['reset']} {msg}")


def warn(msg: str) -> None:
    print(f"  {C['yellow']}!{C['reset']} {msg}")


def fail(msg: str) -> None:
    print(f"  {C['red']}✗{C['reset']} {msg}")


def info(msg: str) -> None:
    print(f"  {C['cyan']}·{C['reset']} {msg}")


def try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def run_cmd(cmd: list[str], timeout: int = 5) -> str | None:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout
        ).strip()
    except Exception:
        return None


# ----------- checks ----------------------------------------------------------


def check_os() -> dict:
    section("1. Système d'exploitation")
    info(f"OS         : {platform.system()} {platform.release()} ({platform.version()})")
    info(f"Architecture: {platform.machine()}")
    info(f"Hostname   : {platform.node()}")
    return {"os": platform.system(), "release": platform.release()}


def check_cpu() -> dict:
    section("2. CPU")
    info(f"Processeur : {platform.processor()}")
    cores_logical = os.cpu_count() or 0
    info(f"Cœurs logiques : {cores_logical}")
    psutil = try_import("psutil")
    if psutil is not None:
        cores_physical = psutil.cpu_count(logical=False) or 0
        info(f"Cœurs physiques : {cores_physical}")
        freq = psutil.cpu_freq()
        if freq:
            info(f"Fréquence : {freq.current:.0f} MHz (max {freq.max:.0f})")
    else:
        warn(
            "psutil non installé — détails fréquence indisponibles. Installe avec: pip install psutil"
        )
    return {"cores_logical": cores_logical}


def check_ram() -> dict:
    section("3. Mémoire vive (RAM)")
    psutil = try_import("psutil")
    if psutil is None:
        warn("psutil non installé — pip install psutil")
        return {}
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024**3)
    avail_gb = vm.available / (1024**3)
    info(f"RAM totale    : {total_gb:.1f} Go")
    info(f"RAM disponible: {avail_gb:.1f} Go")
    if total_gb >= 16:
        ok("16 Go ou plus : confortable pour le développement local et l'inférence.")
    elif total_gb >= 8:
        warn("8-16 Go : suffisant pour le code et l'inférence, mais évite l'entraînement en local.")
    else:
        fail("Moins de 8 Go : entraînement local impossible, inférence à surveiller.")
    return {"total_gb": total_gb, "avail_gb": avail_gb}


def check_disk() -> dict:
    section("4. Disque")
    free_gb = shutil.disk_usage(Path.home()).free / (1024**3)
    info(f"Espace libre sur le disque utilisateur : {free_gb:.1f} Go")
    if free_gb >= 30:
        ok("Espace suffisant pour Docker, MLflow runs, modèles et datasets.")
    elif free_gb >= 10:
        warn("Espace serré : prévoir 10 Go pour Docker + modèles + caches Keras.")
    else:
        fail("Espace insuffisant — il faut au moins 10 Go libres.")
    return {"free_gb": free_gb}


def check_gpu() -> dict:
    section("5. GPU")
    out = run_cmd(["nvidia-smi"])
    if out:
        ok("GPU NVIDIA détectée — sortie nvidia-smi :")
        for line in out.splitlines()[:15]:
            print(f"    {line}")
        return {"nvidia": True, "raw": out}
    warn("Pas de GPU NVIDIA détectée (nvidia-smi indisponible).")
    # On tente WMIC ou PowerShell pour nommer la GPU intégrée
    if platform.system() == "Windows":
        out = run_cmd(["wmic", "path", "win32_VideoController", "get", "Name"])
        if out:
            info("GPU(s) détectée(s) par WMIC :")
            for line in out.splitlines():
                line = line.strip()
                if line and line != "Name":
                    print(f"    - {line}")
    info("Sans GPU NVIDIA : pas d'entraînement local viable. Utiliser Colab pour l'entraînement.")
    return {"nvidia": False}


def check_python() -> dict:
    section("6. Python et environnement")
    info(f"Python : {sys.version.split()[0]} ({sys.executable})")
    info(f"Plateforme : {sys.platform}")
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        ok(f"Environnement virtuel actif : {venv}")
    else:
        warn("Pas d'environnement virtuel détecté (variable VIRTUAL_ENV vide).")
    return {"python": sys.version.split()[0], "venv": venv}


def check_ml_libs() -> dict:
    section("7. Bibliothèques ML installées")
    libs = {
        "tensorflow": "tensorflow",
        "keras": "keras",
        "torch": "torch",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "streamlit": "streamlit",
        "fastapi": "fastapi",
        "mlflow": "mlflow",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime",
    }
    found = {}
    for mod, pip_name in libs.items():
        m = try_import(mod)
        if m is None:
            print(f"  {C['yellow']}-{C['reset']} {pip_name:<14} : non installé")
        else:
            ver = getattr(m, "__version__", "?")
            print(f"  {C['green']}+{C['reset']} {pip_name:<14} : {ver}")
            found[mod] = ver
    return found


def check_tf_devices() -> dict:
    section("8. TensorFlow — détection des devices")
    tf = try_import("tensorflow")
    if tf is None:
        warn("TensorFlow non installé.")
        return {}
    info(f"TensorFlow {tf.__version__}")
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            ok(f"{len(gpus)} GPU détectée(s) par TensorFlow :")
            for g in gpus:
                print(f"    - {g}")
        else:
            warn("Aucune GPU vue par TensorFlow.")
        cpus = tf.config.list_physical_devices("CPU")
        info(f"CPU device(s) : {len(cpus)}")
    except Exception as e:
        fail(f"Erreur TF : {e}")
    return {"tf_version": tf.__version__}


def check_inference_latency() -> dict:
    section("9. Benchmark d'inférence (estimation latence CPU)")
    tf = try_import("tensorflow")
    if tf is None:
        warn("TensorFlow non installé — benchmark sauté.")
        return {}
    np = try_import("numpy")
    if np is None:
        warn("NumPy non installé — benchmark sauté.")
        return {}
    try:
        info("Construction d'un EfficientNetB0 jouet (warmup, peut prendre ~30 s)…")
        from tensorflow.keras.applications import EfficientNetB0

        model = EfficientNetB0(
            weights=None, include_top=True, classes=10, input_shape=(160, 160, 3)
        )
        x = np.random.rand(1, 160, 160, 3).astype("float32") * 255
        # warmup
        for _ in range(3):
            model.predict(x, verbose=0)
        # mesure
        n = 10
        t0 = time.perf_counter()
        for _ in range(n):
            model.predict(x, verbose=0)
        dt = (time.perf_counter() - t0) / n * 1000
        if dt < 100:
            ok(f"Latence moyenne par image : {dt:.1f} ms — excellent pour l'inférence et la démo.")
        elif dt < 300:
            ok(f"Latence moyenne par image : {dt:.1f} ms — correct pour l'inférence et la démo.")
        else:
            warn(f"Latence moyenne par image : {dt:.1f} ms — un peu lent, mais utilisable.")
        return {"latency_ms": dt}
    except Exception as e:
        warn(f"Benchmark non exécuté : {e}")
        return {}


def check_docker() -> dict:
    section("10. Docker (optionnel mais recommandé)")
    out = run_cmd(["docker", "--version"])
    if out:
        ok(out)
        out2 = run_cmd(
            ["docker", "info", "--format", "{{.ServerVersion}} (containers={{.Containers}})"]
        )
        if out2:
            info(f"Docker engine actif : {out2}")
        else:
            warn("Docker installé mais le daemon ne tourne pas.")
        return {"docker": True}
    warn(
        "Docker non installé. Recommandé pour la phase de containerisation. Voir : https://www.docker.com/products/docker-desktop/"
    )
    return {"docker": False}


def check_git() -> dict:
    section("11. Git")
    out = run_cmd(["git", "--version"])
    if out:
        ok(out)
    else:
        fail("Git non installé.")
    return {"git": bool(out)}


# ----------- main ------------------------------------------------------------


def main() -> int:
    print(
        f"\n{C['bold']}╔══════════════════════════════════════════════════════════════════════╗{C['reset']}"
    )
    print(
        f"{C['bold']}║   Diagnostic machine — projet deepvision-cifar10-classifier         ║{C['reset']}"
    )
    print(
        f"{C['bold']}╚══════════════════════════════════════════════════════════════════════╝{C['reset']}"
    )
    print()

    report = {}
    report.update(check_os())
    report.update(check_cpu())
    report.update(check_ram())
    report.update(check_disk())
    report.update(check_gpu())
    report.update(check_python())
    report["libs"] = check_ml_libs()
    report.update(check_tf_devices())
    report.update(check_inference_latency())
    report.update(check_docker())
    report.update(check_git())

    section("12. Verdict synthétique")
    has_gpu = report.get("nvidia", False)
    ram_ok = report.get("total_gb", 0) >= 8
    disk_ok = report.get("free_gb", 0) >= 10
    docker_ok = report.get("docker", False)

    print()
    if has_gpu:
        ok("Tu disposes d'une GPU NVIDIA — entraînement local envisageable.")
    else:
        info("Pas de GPU NVIDIA — on garde Colab pour l'entraînement (recommandé).")
    if ram_ok and disk_ok:
        ok("Capacités suffisantes pour développement local, API, MLflow, Docker.")
    else:
        warn("Capacités juste — privilégier les opérations légères en local.")
    if not docker_ok:
        info("À installer ensuite : Docker Desktop pour la conteneurisation.")
    print(
        f"\n{C['bold']}{C['green']}Diagnostic terminé.{C['reset']} Communique cette sortie à Claude pour ajuster la stratégie.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
