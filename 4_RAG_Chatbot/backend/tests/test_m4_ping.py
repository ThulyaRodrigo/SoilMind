def test_m4_ping(client):
    r = client.get("/api/m4/ping")
    assert r.status_code == 200
    assert r.json() == {"member": "m4", "component": "Localized agricultural chatbot", "status": "ok"}
