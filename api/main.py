from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, time, json
import numpy as np

app = FastAPI(title="Retina DR Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = "/app/models/best_model.h5"
UPLOAD_DIR = "/app/data/uploaded"
METRICS_PATH = "/app/models/eval_metrics.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
START_TIME = time.time()
retraining_status = {"status": "idle", "last_run": None}
_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is None:
        import tensorflow as tf
        _model_cache = tf.keras.models.load_model(MODEL_PATH)
    return _model_cache

def preprocess_bytes(image_bytes):
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME),
        "model_loaded": _model_cache is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    model = load_model()
    img = preprocess_bytes(contents)
    probs = model.predict(img, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    return {
        "grade": predicted_idx,
        "class": CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)}
    }

@app.post("/upload")
async def upload_data(files: list[UploadFile] = File(...), label: int = 0):
    label_dir = os.path.join(UPLOAD_DIR, str(label))
    os.makedirs(label_dir, exist_ok=True)
    saved = []
    for file in files:
        dest = os.path.join(label_dir, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)
    return {"uploaded": len(saved), "files": saved, "label": label}

retraining_status = {"status": "idle", "last_run": None}

def run_retraining():
    global _model_cache
    retraining_status["status"] = "running"
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        model = tf.keras.models.load_model(MODEL_PATH)
        for layer in model.layers[-20:]:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(
            UPLOAD_DIR, target_size=(224,224),
            batch_size=8, class_mode='sparse', subset='training'
        )
        if len(train_gen) > 0:
            model.fit(train_gen, epochs=3, verbose=1)
            model.save(MODEL_PATH)
            _model_cache = None
            retraining_status["status"] = "complete"
        else:
            retraining_status["status"] = "failed - no training data"
    except Exception as e:
        retraining_status["status"] = f"failed: {str(e)}"
    retraining_status["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")

@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    if retraining_status["status"] == "running":
        return {"message": "Retraining already in progress"}
    background_tasks.add_task(run_retraining)
    retraining_status["status"] = "started"
    return {"message": "Retraining triggered"}

@app.get("/retrain/status")
def retrain_status():
    return retraining_status

@app.get("/metrics")
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {"error": "Metrics not found"}
