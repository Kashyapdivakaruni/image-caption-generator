from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
import io
import os
import base64

app = Flask(__name__)

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY', 'YOUR-API-KEY-HERE')}"}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400
    
    file = request.files["image"]
    try:
        # Read and convert image to RGB
        image = Image.open(file.stream).convert("RGB")
        
        # Convert to JPEG bytes
        with io.BytesIO() as buf:
            image.save(buf, format='JPEG')
            image_bytes = buf.getvalue()
        
        # Convert to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Call Hugging Face API
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": image_b64}
        )
        
        if response.status_code == 200:
            result = response.json()
            caption_text = result[0].get('generated_text', '')
            return jsonify({"caption": caption_text})
        else:
            return jsonify({"error": "API request failed", "detail": response.text}), 500
            
    except Exception as e:
        return jsonify({"error": "could not process image", "detail": str(e)}), 400


# Vercel requires "app" to be the handler
app.debug = False
