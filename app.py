from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Lazy-loaded model and processor
model = None
processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model, processor
    if model is None or processor is None:
        # This will download model weights on first run
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400
    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "could not read image", "detail": str(e)}), 400

    load_model()

    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=50)
    caption_text = processor.decode(generated_ids[0], skip_special_tokens=True)

    return jsonify({"caption": caption_text})


if __name__ == "__main__":
    # Run development server
    app.run(host="0.0.0.0", port=5000, debug=True)
