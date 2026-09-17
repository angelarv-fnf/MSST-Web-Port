---
title: MSST Web Port
emoji: 🎵
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: other
---

# MSST Web Port (Server-Side)

Server-side web interface for **Music-Source-Separation-Training** (ZFTurbo v1.0.22).

All model inference runs on the server (Flask + PyTorch). This is **not** a client-side / browser WASM solution.

## How to use

1. Upload an audio file (WAV / FLAC / MP3 / …)
2. Choose a **model type** and matching **config YAML**
3. (Optional) select a checkpoint from the `checkpoints/` folder
4. Click **Separate Sources**
5. Download the stems

## Adding model weights

Place official MSST `.ckpt` / `.pth` files inside the `checkpoints/` directory.  
They will appear automatically in the UI dropdown.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Notes

- GPU (CUDA) is used automatically when available.
- Large models need significant VRAM.
- This Space ships **without** pre-downloaded model weights to keep the image small.
