#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSST Web Port - Server-side Music Source Separation Web Application
Based on Music-Source-Separation-Training v1.0.22 by ZFTurbo

This is a pure server-side Flask application. All model inference runs on the server.
"""

import os
import sys
import uuid
import time
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, send_file,
    send_from_directory, url_for, redirect, flash
)
from werkzeug.utils import secure_filename
import torch
import numpy as np
import soundfile as sf
import librosa

# Make sure local modules are importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from utils.settings import get_model_from_config
from utils.model_utils import demix, prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.audio_utils import normalize_audio, denormalize_audio

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "msst-webport-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"
CONFIG_FOLDER = BASE_DIR / "configs"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
CHECKPOINT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "flac", "mp3", "ogg", "m4a", "aac", "wma"}

# Global model cache (simple, single-model cache)
_model_cache = {
    "model": None,
    "config": None,
    "model_type": None,
    "device": None,
    "checkpoint_path": None,
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_available_configs():
    """Return list of available YAML config files."""
    configs = []
    if CONFIG_FOLDER.exists():
        for p in sorted(CONFIG_FOLDER.rglob("*.yaml")):
            rel = p.relative_to(CONFIG_FOLDER)
            configs.append(str(rel))
    return configs


def list_available_checkpoints():
    """Return list of available checkpoint files (.ckpt, .pth, .pt)."""
    ckpts = []
    if CHECKPOINT_FOLDER.exists():
        for p in sorted(CHECKPOINT_FOLDER.rglob("*")):
            if p.suffix.lower() in {".ckpt", ".pth", ".pt", ".bin"}:
                rel = p.relative_to(CHECKPOINT_FOLDER)
                ckpts.append(str(rel))
    return ckpts


def load_model(model_type: str, config_path: str, checkpoint_path: str = None):
    """Load (or reuse) a model from config + optional checkpoint."""
    global _model_cache

    device = get_device()
    full_config_path = str(CONFIG_FOLDER / config_path)

    # Cache hit?
    if (
        _model_cache["model"] is not None
        and _model_cache["model_type"] == model_type
        and _model_cache["checkpoint_path"] == checkpoint_path
    ):
        return _model_cache["model"], _model_cache["config"], device

    # Load config + model architecture
    model, config = get_model_from_config(model_type, full_config_path)

    # Load weights if provided
    if checkpoint_path:
        full_ckpt = str(CHECKPOINT_FOLDER / checkpoint_path)
        if os.path.isfile(full_ckpt):
            print(f"Loading checkpoint: {full_ckpt}")
            # Use the same loading logic as official inference
            state = torch.load(full_ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                if "state" in state:
                    state = state["state"]
                elif "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]
            # Clean DataParallel prefixes
            new_state = {}
            for k, v in state.items():
                if k.startswith("module."):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
            model.load_state_dict(new_state, strict=False)
        else:
            print(f"WARNING: Checkpoint not found: {full_ckpt}")

    model = model.to(device)
    model.eval()

    _model_cache.update({
        "model": model,
        "config": config,
        "model_type": model_type,
        "device": device,
        "checkpoint_path": checkpoint_path,
    })
    return model, config, device


def run_separation(
    input_path: str,
    output_dir: str,
    model_type: str,
    config_path: str,
    checkpoint_path: str = None,
    target_instrument: str = None,
    use_tta: bool = False,
    extract_instrumental: bool = False,
):
    """
    Run source separation on a single audio file.
    Returns dict with stem paths and metadata.
    """
    model, config, device = load_model(model_type, config_path, checkpoint_path)

    # Read audio
    mix, sr = librosa.load(input_path, sr=None, mono=False)
    if mix.ndim == 1:
        mix = np.stack([mix, mix], axis=0)  # mono -> stereo
    elif mix.shape[0] > 2:
        mix = mix[:2]  # keep only first 2 channels

    # Some models expect specific sample rates
    target_sr = getattr(config.audio, "sample_rate", 44100) if hasattr(config, "audio") else 44100
    if sr != target_sr:
        mix = librosa.resample(mix, orig_sr=sr, target_sr=target_sr, axis=-1)
        sr = target_sr

    # Normalize if configured
    if hasattr(config, "inference") and getattr(config.inference, "normalize", False):
        mix, norm_params = normalize_audio(mix)
    else:
        norm_params = None

    # Prefer target instrument list
    instruments = prefer_target_instrument(config)
    if target_instrument and target_instrument in instruments:
        instruments = [target_instrument]

    # Run demix (core inference from model_utils)
    with torch.no_grad():
        waveforms = demix(config, model, mix, device, model_type=model_type, pbar=True)

    # Optional TTA
    if use_tta:
        waveforms = apply_tta(config, model, mix, waveforms, device, model_type)

    # Denormalize
    if norm_params is not None:
        for instr in waveforms:
            waveforms[instr] = denormalize_audio(waveforms[instr], norm_params)

    # Save stems
    os.makedirs(output_dir, exist_ok=True)
    stem_paths = {}
    base_name = Path(input_path).stem

    for instr, audio in waveforms.items():
        if target_instrument and instr != target_instrument:
            continue
        out_path = os.path.join(output_dir, f"{base_name}_{instr}.wav")
        # Ensure shape (channels, samples)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        sf.write(out_path, audio.T, sr, subtype="FLOAT")
        stem_paths[instr] = out_path

    # Optional instrumental (mixture - vocals)
    if extract_instrumental and "vocals" in waveforms:
        instr_audio = mix - waveforms["vocals"]
        out_path = os.path.join(output_dir, f"{base_name}_instrumental.wav")
        sf.write(out_path, instr_audio.T, sr, subtype="FLOAT")
        stem_paths["instrumental"] = out_path

    return {
        "stems": stem_paths,
        "sample_rate": sr,
        "instruments": list(stem_paths.keys()),
        "duration": mix.shape[-1] / sr,
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    configs = list_available_configs()
    checkpoints = list_available_checkpoints()
    device = str(get_device())
    return render_template(
        "index.html",
        configs=configs,
        checkpoints=checkpoints,
        device=device,
        model_types=[
            "mdx23c", "htdemucs", "bs_roformer", "mel_band_roformer",
            "scnet", "scnet_tran", "scnet_masked", "segm_models",
            "torchseg", "bandit", "bandit_v2", "bs_mamba2",
            "mel_band_conformer", "conformer", "apollo", "dttnet",
        ],
    )


@app.route("/api/configs")
def api_configs():
    return jsonify(list_available_configs())


@app.route("/api/checkpoints")
def api_checkpoints():
    return jsonify(list_available_checkpoints())


@app.route("/api/separate", methods=["POST"])
def api_separate():
    """Main separation endpoint."""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}"}), 400

        model_type = request.form.get("model_type", "mdx23c")
        config_path = request.form.get("config_path", "")
        checkpoint_path = request.form.get("checkpoint_path", "") or None
        target_instrument = request.form.get("target_instrument", "") or None
        use_tta = request.form.get("use_tta", "false").lower() == "true"
        extract_instrumental = request.form.get("extract_instrumental", "false").lower() == "true"

        if not config_path:
            return jsonify({"error": "config_path is required"}), 400

        # Save upload
        job_id = str(uuid.uuid4())[:8]
        upload_dir = UPLOAD_FOLDER / job_id
        output_dir = OUTPUT_FOLDER / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = secure_filename(file.filename)
        input_path = str(upload_dir / filename)
        file.save(input_path)

        # Run separation
        start = time.time()
        result = run_separation(
            input_path=input_path,
            output_dir=str(output_dir),
            model_type=model_type,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            target_instrument=target_instrument,
            use_tta=use_tta,
            extract_instrumental=extract_instrumental,
        )
        elapsed = time.time() - start

        # Build download URLs
        stems = {}
        for instr, path in result["stems"].items():
            stems[instr] = {
                "filename": os.path.basename(path),
                "url": f"/download/{job_id}/{os.path.basename(path)}",
            }

        return jsonify({
            "success": True,
            "job_id": job_id,
            "stems": stems,
            "instruments": result["instruments"],
            "sample_rate": result["sample_rate"],
            "duration": round(result["duration"], 2),
            "elapsed_seconds": round(elapsed, 2),
            "device": str(get_device()),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/download/<job_id>/<filename>")
def download_stem(job_id, filename):
    directory = OUTPUT_FOLDER / job_id
    return send_from_directory(directory, filename, as_attachment=True)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    })


@app.route("/api/clear_cache", methods=["POST"])
def clear_cache():
    global _model_cache
    if _model_cache["model"] is not None:
        del _model_cache["model"]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    _model_cache = {
        "model": None, "config": None, "model_type": None,
        "device": None, "checkpoint_path": None,
    }
    return jsonify({"status": "cache cleared"})


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"Starting MSST Web Port on http://{host}:{port}")
    print(f"Device: {get_device()}")
    print(f"Configs found: {len(list_available_configs())}")
    print(f"Checkpoints found: {len(list_available_checkpoints())}")
    app.run(host=host, port=port, debug=debug, threaded=True)
