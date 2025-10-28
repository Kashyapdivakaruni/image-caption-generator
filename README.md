# Image Caption Generator (BLIP + Flask)

This small project runs a web app that generates an image caption when a user uploads an image. It uses the pre-trained BLIP model from Hugging Face: `Salesforce/blip-image-captioning-base` — no training required.

It is optimized for local CPU use (will work on GPU if available).

## What you need
- Python 3.8+ on Windows
- PowerShell (instructions below)
- Internet access (first run downloads model weights ~100-500MB)

## Setup (PowerShell)

# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install a CPU PyTorch wheel (official wheels). Pick the correct command for your Python version.
# On Windows, for CPU-only, run:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install other Python packages
pip install -r requirements.txt

## Run the app
python app.py

Open http://127.0.0.1:5000 in your browser and upload an image. The app will return a generated caption.

## Quick test (script)
You can run the small test which downloads a sample image and prints a caption:
python test_caption.py

## Notes
- First run will download the model weights (some hundred MB). Be patient.
- If you have a GPU and CUDA installed, PyTorch will use it automatically.
- For production or larger throughput, run behind Gunicorn/Uvicorn and add batching or caching.

## Files
- `app.py` — Flask web app and model integration
- `templates/index.html` — upload form
- `requirements.txt` — required Python packages (except torch)
- `test_caption.py` — script to run a single caption generation from a sample image

If you want, I can add Docker support or create a standalone executable next.
