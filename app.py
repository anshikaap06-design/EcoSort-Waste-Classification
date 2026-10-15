
import json
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MODEL_DIR = Path("model")
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "waste_model.keras"
labels_path = MODEL_DIR / "class_names.json"

if not model_path.exists() or not labels_path.exists():
    raise FileNotFoundError(
        "Trained model missing. Check the model folder."
    )

model = tf.keras.models.load_model(model_path)

with open(labels_path, "r") as f:
    class_names = json.load(f)

advice = {
    "cardboard": "Keep it clean and dry for cardboard recycling.",
    "glass": "Separate glass and follow local recycling guidelines.",
    "metal": "Keep metal items separate for recycling.",
    "paper": "Keep paper dry and send it for paper recycling.",
    "plastic": "Check local recycling rules and clean the container.",
    "trash": "Use the appropriate local general-waste collection."
}


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    image_url = None

    if request.method == "POST":
        uploaded = request.files.get("image")

        if not uploaded or not uploaded.filename:
            error = "Please select an image."
        else:
            try:
                image = Image.open(uploaded.stream).convert("RGB")
                image = image.resize((224, 224))

                array = np.asarray(image, dtype=np.float32)
                array = np.expand_dims(array, axis=0)

                predictions = model.predict(array, verbose=0)[0]
                index = int(np.argmax(predictions))
                category = class_names[index]

                confidence = float(predictions[index]) * 100

                filename = f"{uuid.uuid4().hex}.jpg"
                image.save(UPLOAD_DIR / filename, format="JPEG")
                image_url = f"/static/uploads/{filename}"

                result = {
                    "category": category.replace("_", " ").title(),
                    "confidence": round(confidence, 2),
                    "advice": advice.get(
                        category.lower(),
                        "Check local disposal guidelines."
                    )
                }

            except (UnidentifiedImageError, OSError, ValueError):
                error = "Invalid image. Please upload a JPG or PNG."

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_url=image_url
    )


@app.errorhandler(413)
def too_large(error):
    return (
        "Image is too large. Please upload an image under 5 MB.",
        413
    )


if __name__ == "__main__":
    app.run(debug=True)