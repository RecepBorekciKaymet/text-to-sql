import pytest
from fastapi.testclient import TestClient
from main import app  # Import FastAPI app

client = TestClient(app)

def test_generate_sql():
    response = client.post("/generate-sql", json={"text": "Show all women’s sports shoes"})
    assert response.status_code == 200
    assert "sql_query" in response.json()

def test_generate_sql_empty():
    response = client.post("/generate-sql", json={"text": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty"

def test_execute_sql():
    response = client.post("/execute-sql", json={"query": "SELECT * FROM Products LIMIT 1"})
    assert response.status_code == 200
    assert "query_result" in response.json()

def test_blocked_sql_execution():
    response = client.post("/execute-sql", json={"query": "DROP TABLE Products"})
    assert response.status_code == 200  # Still returns success but contains an error message
    assert "error" in response.json()["query_result"].lower()  # Ensures it works regardless of case

