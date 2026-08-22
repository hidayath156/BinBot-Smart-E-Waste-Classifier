import os
import pickle
import numpy as np
import uuid

from ai_fallback import ai_detect_e_waste, ai_explain_known_e_waste
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/ewaste_efficientnetb0_final.h5"
CLASS_INDEX_PATH = "class_indices.pkl"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 55

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "rb") as f:
    class_indices = pickle.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = np.argmax(preds)
    confidence = preds[idx] * 100
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

        if file and file.filename != "":
            filename = str(uuid.uuid4()) + "_" + file.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            prediction, confidence = predict_image(image_path)

            if prediction == "Unknown / Low Confidence":
                ai_result = ai_detect_e_waste(image_path)
            else:
                ai_result = ai_explain_known_e_waste(prediction)

            print("AI RESULT:", ai_result)  # 🔍 DEBUG

            # Dynamic scores
            try:
                recycle_score = int(ai_result.get("recyclability_score"))
            except:
                recycle_score = 70

            try:
                impact_score = int(ai_result.get("impact_score"))
            except:
                impact_score = 70

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        ai_result=ai_result,
        recycle_score=recycle_score,
        impact_score=impact_score
    )

if __name__ == "__main__":
    app.run(debug=True)