'''
POST request to API
'''

import requests

data = {"age": 28,
        "workclass": "Private",
        "education": "Prof-school",
        "education_num": 15,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 55,
        "native_country": "United-States"
        }

r = requests.post('http://127.0.0.1:8000/predict',
                  json=data)
print(f"Request Status Code: {r.status_code}")
print(f"Request Response (Model Inference Result): {r.json()}")
