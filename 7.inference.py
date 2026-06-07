import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "columns": [
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age"
    ],
    "data": [[0.639947, 0.866045, -0.031990, 0.670643,
              -0.181541, 0.166619, 0.468492, 1.425995]]
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Prediction:", response.json())