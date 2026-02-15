from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json




app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/leaf_disease_model.h5")
with open("model/class_indices.json") as f:
    class_indices = json.load(f)

# Class labels (MUST match training folder names)
# class_labels = [
#     "Apple___Healthy",
#     "Apple___Scab",
#     "Apple___Black_rot",
#     "Potato___Healthy",
#     "Potato___Early_blight",
#     "Potato___Late_blight",
#     "Tomato___Healthy",
#     "Tomato___Early_blight",
#     "Tomato___Late_blight"
# ]



# Reverse dictionary
class_labels = list(class_indices.keys())

treatment_dict = {

    # Apple
    "Apple___Healthy": "No treatment required. Maintain proper sunlight and irrigation.",
    "Apple___Scab": "Apply fungicide like captan or sulfur spray.",
    "Apple___Black_rot": "Prune infected branches and apply copper fungicide.",

    # Grape
    "Grape___Healthy": "No disease detected. Maintain vineyard hygiene.",
    "Grape___Black_rot": "Remove infected leaves and apply myclobutanil fungicide.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood and apply protective fungicides.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Use copper-based fungicide and ensure airflow.",

    # Strawberry
    "Strawberry___Healthy": "Plant is healthy. Maintain proper soil moisture.",
    "Strawberry___Leaf_scorch": "Remove infected leaves and apply fungicide like captan."
}



def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)

    result = class_labels[class_index]
    crop, disease = result.split("___")
    treatment = treatment_dict.get(result,"Treatment information not available.")

    return crop, disease, confidence, treatment

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("static/uploads", file.filename)
            file.save(filepath)

            crop, disease, confidence, treatment = predict_image(filepath)


            return render_template("index.html",
                                   crop=crop,
                                   disease=disease,
                                   confidence=confidence,
                                   treatment=treatment,
                                   image_path=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
