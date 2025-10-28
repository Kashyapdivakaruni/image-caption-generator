from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
import io
import os
import base64
import json

app = Flask(__name__)

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
API_KEY = os.environ.get('HUGGINGFACE_API_KEY')

if not API_KEY:
    print("Warning: HUGGINGFACE_API_KEY environment variable is not set")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the service is running"""
    return jsonify({"status": "healthy", "api_key_configured": bool(API_KEY)}), 200

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400
    
    file = request.files["image"]
    
    try:
        # Verify file content
        if not file.content_type.startswith('image/'):
            return jsonify({"error": "invalid file type, must be an image"}), 400
        
        # Read and convert image to RGB
        image = Image.open(file.stream).convert("RGB")
        
        # Resize if too large (max 1000x1000)
        max_size = 1000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to JPEG bytes
        with io.BytesIO() as buf:
            image.save(buf, format='JPEG', quality=85)
            image_bytes = buf.getvalue()
        
        # Convert to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Call Hugging Face API with timeout
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": image_b64},
            timeout=25  # Vercel has 30s timeout, we use 25s to be safe
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                caption_text = result[0].get('generated_text', '')
                return jsonify({"caption": caption_text})
            else:
                return jsonify({"error": "unexpected API response format", "detail": str(result)}), 500
        else:
            error_detail = str(response.text)
            if response.status_code == 503:
                error_detail = "Model is loading, please try again in a few seconds"
            return jsonify({"error": "API request failed", "detail": error_detail}), response.status_code
            
    except requests.Timeout:
        return jsonify({"error": "request timeout", "detail": "The image captioning service took too long to respond"}), 504
    except Exception as e:
        return jsonify({"error": "could not process image", "detail": str(e)}), 400


# Vercel requires "app" to be the handler
app.debug = False
