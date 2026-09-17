#!/bin/bash
# Quick launcher for MSST Web Port
export PORT=${PORT:-7860}
export HOST=${HOST:-0.0.0.0}
python app.py
