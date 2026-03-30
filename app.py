import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(input_path)

    results = model.predict(
        source=input_path,
        conf=0.20,   
        save=False,
        show=False
    )

    boxes = results[0].boxes

    if len(boxes) == 0:
        detected_text = "No objects detected"
        top_conf = 0.0
    else:
        confidences = boxes.conf.tolist()
        class_ids = boxes.cls.tolist()

        best_index = confidences.index(max(confidences))
        best_class_id = int(class_ids[best_index])
        detected_text = model.names[best_class_id]
        top_conf = float(confidences[best_index])

    return render_template(
        "result.html",
        uploaded_image=f"static/uploads/{filename}",
        detected_text=detected_text,
        top_conf=round(top_conf, 2)
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)