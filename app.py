import time
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import uvicorn

# =========================================================
# LOAD MODEL
# =========================================================
model_path = "mlartifacts/260170601487467980/674f633fbf794e31bb4638ba86a1a0d0/artifacts/model"
model = mlflow.sklearn.load_model(f"E:/SMSML_Lilis Aprilia/Membangun_model/{model_path}")

# =========================================================
# PROMETHEUS METRICS
# =========================================================
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
SUCCESS_COUNT = Counter('prediction_success_total', 'Total successful predictions')
FAILED_COUNT = Counter('prediction_failed_total', 'Total failed predictions')
POSITIVE_COUNT = Counter('prediction_positive_total', 'Total positive predictions (Diabetes)')
NEGATIVE_COUNT = Counter('prediction_negative_total', 'Total negative predictions (Non-Diabetes)')
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy')
MODEL_PRECISION = Gauge('model_precision', 'Model precision')
MODEL_RECALL = Gauge('model_recall', 'Model recall')
MODEL_F1 = Gauge('model_f1_score', 'Model F1 score')

# Set static metrics
MODEL_ACCURACY.set(0.7792)
MODEL_PRECISION.set(0.7273)
MODEL_RECALL.set(0.5926)
MODEL_F1.set(0.6531)

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    REQUEST_COUNT.inc()
    start = time.time()
    try:
        body = await request.json()
        data = pd.DataFrame(body["data"], columns=body["columns"])
        prediction = model.predict(data)
        latency = time.time() - start
        LATENCY.observe(latency)
        SUCCESS_COUNT.inc()
        if prediction[0] == 1:
            POSITIVE_COUNT.inc()
        else:
            NEGATIVE_COUNT.inc()
        return {"prediction": prediction.tolist(), "latency": latency}
    except Exception as e:
        FAILED_COUNT.inc()
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)