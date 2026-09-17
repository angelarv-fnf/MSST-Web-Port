# MSST Web Port (Server-Side)

A **server-side** HTML/Flask web interface for [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) v1.0.22 by ZFTurbo.

This is **not** a client-side (browser WASM / ONNX) solution. All model loading, STFT, demixing and inference run on the Python server.

## Features

- Upload audio (WAV, FLAC, MP3, OGG, M4A…)
- Select model type + YAML config
- Optional checkpoint loading
- Optional Test-Time Augmentation (TTA)
- Optional instrumental extraction
- Download separated stems as WAV
- Model caching for faster subsequent runs
- Clean modern dark UI

## Directory Layout

```
msst_webport/
├── app.py                 # Flask application (main entry point)
├── inference.py           # Original MSST inference helpers
├── requirements.txt       # Original MSST deps
├── requirements-web.txt   # Extra web deps (Flask etc.)
├── configs/               # All YAML model configs
├── models/                # Model architectures
├── utils/                 # metrics, model_utils, audio_utils, settings…
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
├── checkpoints/           # ← PUT YOUR .ckpt / .pth FILES HERE
├── uploads/               # Temporary uploads
└── outputs/               # Generated stems
```

## Quick Start

```bash
# 1. Create virtualenv (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install PyTorch (choose the right CUDA / CPU build)
#    See https://pytorch.org/get-started/locally/
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining deps
pip install -r requirements-web.txt
# (or also: pip install -r requirements.txt)

# 4. Place model weights
#    Download any official MSST checkpoint and put it in checkpoints/
#    Example:
#    checkpoints/model_bs_roformer.ckpt

# 5. Run the server
python app.py
# → http://0.0.0.0:7860
```

Environment variables:

| Variable | Default     | Description              |
|----------|-------------|--------------------------|
| `PORT`   | `7860`      | HTTP port                |
| `HOST`   | `0.0.0.0`   | Bind address             |
| `DEBUG`  | `false`     | Flask debug mode         |
| `SECRET_KEY` | (random) | Flask session secret |

## Usage Notes

1. **Configs** – All YAML files from the original repo are under `configs/`. Pick the one that matches your model architecture and training setup.
2. **Checkpoints** – The web UI lists every `.ckpt` / `.pth` / `.pt` found under `checkpoints/`. Without a checkpoint the model runs with random weights (useless for real separation).
3. **Model types** – Must match the architecture used when the config/checkpoint was created (`mdx23c`, `bs_roformer`, `mel_band_roformer`, `htdemucs`, `scnet`, …).
4. **GPU** – Automatically uses CUDA if available, otherwise MPS (Apple Silicon) or CPU.
5. **Memory** – Large models + long audio can consume a lot of VRAM/RAM. Use shorter clips for testing.

## API Endpoints

| Method | Path                  | Description                          |
|--------|-----------------------|--------------------------------------|
| GET    | `/`                   | Main UI                              |
| POST   | `/api/separate`       | Run separation (multipart form)      |
| GET    | `/download/<job>/<file>` | Download a stem                   |
| GET    | `/api/health`         | Health + device info                 |
| GET    | `/api/configs`        | List available configs               |
| GET    | `/api/checkpoints`    | List available checkpoints           |
| POST   | `/api/clear_cache`    | Unload current model from memory     |

## Relationship to original MSST

This port re-uses the official:

- `utils/model_utils.py` → `demix`, `apply_tta`, …
- `utils/settings.py` → `get_model_from_config`
- `utils/audio_utils.py`
- All model definitions under `models/`
- All configs under `configs/`

Only a thin Flask + HTML/JS layer was added on top. No client-side inference is performed.

## License

The original Music-Source-Separation-Training code retains its license (see `LICENSE` / README of the upstream repo). This web port wrapper is provided as-is for research and personal use.
