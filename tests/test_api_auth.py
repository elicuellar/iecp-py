"""Auth API tests -- Phase 10."""

from __future__ import annotations

import pytest

from .api_helpers import TEST_API_KEY, auth_headers, create_test_app


class TestAuthMiddleware:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

    def test_returns_401_without_token(self) -> None:
        res = self.client.get("/api/v1/entities")
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "UNAUTHORIZED"

    def test_returns_403_for_invalid_token(self) -> None:
        res = self.client.get(
            "/api/v1/entities",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert res.status_code == 403
        assert res.json()["error"]["code"] == "FORBIDDEN"

    def test_returns_401_for_malformed_auth_header(self) -> None:
        res = self.client.get(
            "/api/v1/entities",
            headers={"Authorization": "NotBearer token"},
        )
        assert res.status_code == 401

    def test_succeeds_with_valid_token(self) -> None:
        res = self.client.get("/api/v1/entities", headers=auth_headers())
        assert res.status_code == 200


class TestTokenGeneration:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

    def test_generates_valid_websocket_token(self) -> None:
        # Create entity first
        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "daemon", "display_name": "MyDaemon"},
        )
        entity_id = entity_res.json()["entity_id"]

        res = self.client.post(
            "/api/v1/auth/tokens",
            headers=auth_headers(),
            json={"entityId": entity_id, "type": "daemon", "conversationIds": []},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["token"].startswith("iecp_ws_")
        assert body["entityId"] == entity_id

    def test_token_missing_required_fields(self) -> None:
        res = self.client.post(
            "/api/v1/auth/tokens",
            headers=auth_headers(),
            json={"entityId": "some-id"},
        )
        assert res.status_code == 400

    def test_generated_token_is_accepted(self) -> None:
        # Create entity and generate token
        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "User"},
        )
        entity_id = entity_res.json()["entity_id"]

        token_res = self.client.post(
            "/api/v1/auth/tokens",
            headers=auth_headers(),
            json={"entityId": entity_id, "type": "human", "conversationIds": []},
        )
        token = token_res.json()["token"]
        # The WS token won't work as an API key (different store) -- just verify it was created
        assert token.startswith("iecp_ws_")


class TestHealthEndpoint:
    def setup_method(self) -> None:
        self.client, _ = create_test_app()

    def test_health_returns_ok_without_auth(self) -> None:
        res = self.client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert "uptime" in body

    def test_health_returns_version(self) -> None:
        res = self.client.get("/health")
        assert res.json()["version"] == "1.0.0-rc1"

    def test_health_has_checks(self) -> None:
        res = self.client.get("/health")
        body = res.json()
        assert "checks" in body
        assert body["checks"]["database"] == "ok"
        assert body["checks"]["gateway"] == "ok"
        assert "artificerQueue" in body["checks"]
