"""
Quick script to download a sample image and run the BLIP model locally to print a caption.
Run: python test_caption.py
"""
import requests
from PIL import Image
import io
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def download_sample():
    url = "https://images.unsplash.com/photo-1519681393784-d120267933ba"
    r = requests.get(url + "?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=60")
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def caption_image(image: Image.Image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=50)
    caption_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption_text


def main():
    print("Downloading sample image...")
    img = download_sample()
    print("Generating caption (this will download model weights if not present)...")
    caption = caption_image(img)
    print("Caption:", caption)


if __name__ == "__main__":
    main()
