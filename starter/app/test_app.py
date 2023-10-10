'''
Test cases for FastAPI app
'''

from fastapi.testclient import TestClient
from main import app
import json
import pytest


client = TestClient(app)


def test_get_method():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


@pytest.fixture(scope='module')
def low_salary_data():
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

    return data

@pytest.fixture(scope='module')
def high_salary_data():
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

    return data

def test_low_salary(low_salary_data):
    '''
    Tests the "/predict" path of the API to return '0'

    Inputs
    ------
    low_salary_data : dict
        Sample dataset that retuens '0' as prediction.
    '''
    data = json.dumps(low_salary_data)
    r = client.post("/predict", data=data)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}

def test_high_salary(high_salary_data):
    '''
    Tests the "/predict" path of the API to return '1'

    Inputs
    ------
    high_salary_data : dict
        Sample dataset that returns '1' as prediction.
    '''
    data = json.dumps(high_salary_data)
    r = client.post("/predict", data=data)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
