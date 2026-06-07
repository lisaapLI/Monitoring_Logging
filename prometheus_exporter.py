import time
import random
import requests
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# =========================================================
# METRICS DEFINITION
# =========================================================

# 1. Accuracy
model_accuracy = Gauge('model_accuracy', 'Accuracy of the ML model')

# 2. Precision
model_precision = Gauge('model_precision', 'Precision of the ML model')

# 3. Recall
model_recall = Gauge('model_recall', 'Recall of the ML model')

# 4. F1 Score
model_f1_score = Gauge('model_f1_score', 'F1 Score of the ML model')

# 5. Prediction Latency
prediction_latency = Histogram('prediction_latency_seconds', 'Latency of predictions in seconds')

# 6. Total Predictions
total_predictions = Counter('total_predictions', 'Total number of predictions made')

# 7. Failed Predictions
failed_predictions = Counter('failed_predictions', 'Total number of failed predictions')

# 8. Positive Predictions (Diabetes)
positive_predictions = Counter('positive_predictions', 'Total predictions classified as Diabetes')

# 9. Negative Predictions (Non-Diabetes)
negative_predictions = Counter('negative_predictions', 'Total predictions classified as Non-Diabetes')

# 10. Model Version
model_version = Gauge('model_version', 'Current model version')

# =========================================================
# SIMULATE METRICS
# =========================================================

def simulate_metrics():
    # Set static metrics
    model_accuracy.set(0.7792)
    model_precision.set(0.7273)
    model_recall.set(0.5926)
    model_f1_score.set(0.6531)
    model_version.set(1)

    while True:
        # Simulate prediction
        latency = random.uniform(0.01, 0.5)
        prediction_latency.observe(latency)
        total_predictions.inc()

        # Simulate success/failure
        if random.random() > 0.05:
            if random.random() > 0.5:
                positive_predictions.inc()
            else:
                negative_predictions.inc()
        else:
            failed_predictions.inc()

        time.sleep(1)

if __name__ == '__main__':
    print("Starting Prometheus exporter on port 8000...")
    start_http_server(8000)
    simulate_metrics()