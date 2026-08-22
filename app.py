import os
import pickle
import uuid

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, render_template, request

from ai_fallback import (
    ai_detect_e_waste,
    ai_explain_known_e_waste
)


UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/ewaste_efficientnetb0_final.tflite"
CLASS_INDEX_PATH = "class_indices.pkl"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 55
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=1
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


with open(CLASS_INDEX_PATH, "rb") as f:
    class_indices = pickle.load(f)

idx_to_class = {
    v: k for k, v in class_indices.items()
}


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(img, dtype=np.float32)

    # EfficientNet Keras preprocessing is built into the model.
    # Do not normalize to [-1, 1].
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(
        input_details[0]["index"],
        arr
    )

    interpreter.invoke()

    preds = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    idx = int(np.argmax(preds))

    confidence = float(preds[idx]) * 100

    label = idx_to_class[idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown / Low Confidence", confidence

    return label, confidence


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_path = None
    ai_result = None
    recycle_score = None
    impact_score = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file or not file.filename:

            return render_template(
                "index.html",
                prediction="Please select an image.",
                confidence=None,
                image_path=None,
                ai_result=None,
                recycle_score=None,
                impact_score=None
            )

        if not allowed_file(file.filename):

            return render_template(
                "index.html",
                prediction="Invalid image format.",
                confidence=None,
                image_path=None,
                ai_result=None,
                recycle_score=None,
                impact_score=None
            )

        ext = os.path.splitext(
            file.filename
        )[1].lower()

        filename = f"{uuid.uuid4().hex}{ext}"

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        try:

            file.save(image_path)

            prediction, confidence = predict_image(
                image_path
            )

            if prediction == "Unknown / Low Confidence":

                ai_result = ai_detect_e_waste(
                    image_path
                )

            else:

                ai_result = ai_explain_known_e_waste(
                    prediction
                )

            try:

                recycle_score = int(
                    ai_result.get(
                        "recyclability_score"
                    )
                )

            except (
                TypeError,
                ValueError,
                AttributeError
            ):

                recycle_score = 70

            try:

                impact_score = int(
                    ai_result.get(
                        "impact_score"
                    )
                )

            except (
                TypeError,
                ValueError,
                AttributeError
            ):

                impact_score = 70

        except Exception as e:

            print(
                "Prediction error:",
                str(e)
            )

            prediction = "Unable to process image."
            confidence = None
            ai_result = None
            recycle_score = None
            impact_score = None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        ai_result=ai_result,
        recycle_score=recycle_score,
        impact_score=impact_score
    )


@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        prediction="Image is too large. Maximum size is 10 MB.",
        confidence=None,
        image_path=None,
        ai_result=None,
        recycle_score=None,
        impact_score=None
    ), 413


@app.route("/health")
def health():

    return {
        "status": "ok"
    }


if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )