from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.main import app


def test_root_serves_chat_ui():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "House Price SQL Agent" in response.text
    assert 'id="chat-form"' in response.text


def test_trace_view_serves_readable_viewer():
    client = TestClient(app)
    trace_id = "trace_658d99e5"

    response = client.get(f"/traces/{trace_id}/view")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Execution timeline" in response.text
    assert "SQL evidence" in response.text


def test_chat_reports_database_connection_error(monkeypatch):
    def fail_database_connection(*args):
        raise OperationalError("connect", {}, Exception("authentication failed"))

    monkeypatch.setattr("app.api.routes.run_agent", fail_database_connection)
    response = TestClient(app).post(
        "/chat",
        json={"user_id": "user_api", "conversation_id": "conv_api", "message": "How many houses are there?"},
    )

    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_health_and_chat_and_trace():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    chat = client.post(
        "/chat",
        json={"user_id": "user_api", "conversation_id": "conv_api", "message": "How many 3-bedroom houses are there?"},
    )
    assert chat.status_code == 200
    body = chat.json()
    assert "trace_id" in body
    assert body["conversation_id"] == "conv_api"
    assert "answer" in body

    trace = client.get(f"/traces/{body['trace_id']}")
    assert trace.status_code == 200
    assert trace.json()["trace_id"] == body["trace_id"]
