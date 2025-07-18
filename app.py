from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model
from ultralytics import YOLO  # Make sure ultralytics is installed: pip install ultralytics

app = Flask(__name__)
UPLOAD_FOLDER = "../uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load all models
crop_model = load_model("../ai_models/plant_disease_model.h5")
weed_model = load_model("../ai_models/weed_segmentor.h5")  # Replace with your .h5 path
pest_model = YOLO("yolov8n.pt")  # Downloads automatically if not present

# Labels
crop_labels = ['Tomato_Healthy', 'Tomato_Bacterial_spot', 'Tomato_Leaf_Mold', 'Tomato_Yellow_Leaf_Curl']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    if 'task' not in request.form:
        return jsonify({"error": "No task provided (crop/pest/weed)"}), 400

    task = request.form['task']
    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    if task == "crop":
        img = Image.open(image_path).resize((28, 28))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        preds = crop_model.predict(img)
        predicted_class = crop_labels[np.argmax(preds)]
        return jsonify({"task": "crop", "prediction": predicted_class})

    elif task == "pest":
        results = pest_model(image_path)
        pred_boxes = results[0].boxes
        pest_data = []
        for box in pred_boxes:
            class_id = int(box.cls[0].item())
            confidence = round(box.conf[0].item(), 2)
            pest_data.append({"class_id": class_id, "confidence": confidence})
        return jsonify({"task": "pest", "detections": pest_data})

    elif task == "weed":
        img = Image.open(image_path).resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        preds = weed_model.predict(img)
        mean_pred = np.mean(preds)
        weed_level = "high" if mean_pred > 0.5 else "low"
        return jsonify({"task": "weed", "weed_level": weed_level, "raw_score": float(mean_pred)})

    else:
        return jsonify({"error": "Unknown task. Use crop, pest, or weed."}), 400

if __name__ == '__main__':
    app.run(debug=True)
