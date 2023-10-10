'''
POST request to API
'''

import requests

data = {"age": 53,
        "workclass": "Private",
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Other-service",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "Germany"
        }


r = requests.post('http://127.0.0.1:8000/predict',
                  json=data)
print(f"Request Status Code: {r.status_code}")
print(f"Request Response (Model Inference Result): {r.json()}")
