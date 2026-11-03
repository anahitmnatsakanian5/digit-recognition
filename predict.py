"""Predict a digit from an image file.

Usage: python predict.py path/to/digit.png
The image should be a single digit; dark digit on light background or the
reverse both work (it is auto-inverted to match MNIST's white-on-black).
"""
import sys
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras


def load_image(path):
    img = Image.open(path).convert("L")
    # MNIST is white digit on black background; invert if the image is mostly light
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    return (np.array(img) / 255.0).astype("float32").reshape(1, 28, 28, 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py path/to/digit.png")
    model = keras.models.load_model("model.keras")
    probs = model.predict(load_image(sys.argv[1]), verbose=0)[0]
    digit = int(probs.argmax())
    print(f"Predicted digit: {digit} (confidence {probs[digit]:.1%})")
