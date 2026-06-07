import requests
import json

# =========================================================
# INFERENCE - Kirim data ke model serving
# =========================================================

url = "http://127.0.0.1:5001/invocations"

data = {
    "dataframe_split": {
        "columns": [
            "Pregnancies", "Glucose", "BloodPressure", 
            "SkinThickness", "Insulin", "BMI", 
            "DiabetesPedigreeFunction", "Age"
        ],
        "data": [[0.639947, 0.866045, -0.031990, 0.670643, 
                  -0.181541, 0.166619, 0.468492, 1.425995]]
    }
}

headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print("Status Code:", response.status_code)
    print("Prediction:", response.json())
except Exception as e:
    print("Error:", e)