from setuptools import setup

setup(
    name="image-caption-generator",
    version="1.0.0",
    description="A web application that generates image captions using BLIP model",
    python_requires=">=3.9,<3.12",
    install_requires=[
        "flask>=2.0.1",
        "transformers>=4.21.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=8.3.1",
        "requests>=2.26.0",
        "gunicorn>=20.1.0",
    ],
)