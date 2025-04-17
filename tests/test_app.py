# tests/test_app.py
from fastapi.testclient import TestClient
from src.integration.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Medical Assistant Application!"}

def test_db_connection():
    response = client.get("/db-test")
    assert response.status_code == 200
    json_response = response.json()
    assert "Connected to DB" in json_response["message"]
    
def test_openai_integration():
    response = client.post("/openai-test", json={"prompt": "Hello, how can I assist you today?"})
    assert response.status_code == 200
    json_response = response.json()
    assert "response" in json_response
    
def test_whatsapp_integration():
    response = client.post("/whatsapp-test", json={"Body": "Hello, how can I assist you today?"})
    assert response.status_code == 200
    json_response = response.json()
    assert "message" in json_response
