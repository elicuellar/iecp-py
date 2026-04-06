"""Artificer API tests -- Phase 10."""

from __future__ import annotations

import pytest

from .api_helpers import auth_headers, create_test_app


class TestArtificerRoutes:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot"},
        )
        self.entity_id = entity_res.json()["entity_id"]

    def test_registers_artificer(self) -> None:
        res = self.client.post(
            "/api/v1/artificers",
            headers=auth_headers(),
            json={"entityId": self.entity_id, "persona": "Helpful assistant"},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["entityId"] == self.entity_id
        assert body["persona"] == "Helpful assistant"
        assert "registeredAt" in body

    def test_requires_entity_id(self) -> None:
        res = self.client.post(
            "/api/v1/artificers",
            headers=auth_headers(),
            json={},
        )
        assert res.status_code == 400

    def test_unregisters_artificer(self) -> None:
        self.client.post(
            "/api/v1/artificers",
            headers=auth_headers(),
            json={"entityId": self.entity_id},
        )

        res = self.client.delete(
            f"/api/v1/artificers/{self.entity_id}",
            headers=auth_headers(),
        )
        assert res.status_code == 204

        # Verify it's gone
        list_res = self.client.get("/api/v1/artificers", headers=auth_headers())
        assert len(list_res.json()) == 0

    def test_returns_404_for_unregistered_artificer_delete(self) -> None:
        res = self.client.delete(
            "/api/v1/artificers/nonexistent",
            headers=auth_headers(),
        )
        assert res.status_code == 404

    def test_lists_registered_artificers(self) -> None:
        entity2_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot2"},
        )
        entity2_id = entity2_res.json()["entity_id"]

        self.client.post(
            "/api/v1/artificers",
            headers=auth_headers(),
            json={"entityId": self.entity_id},
        )
        self.client.post(
            "/api/v1/artificers",
            headers=auth_headers(),
            json={"entityId": entity2_id},
        )

        res = self.client.get("/api/v1/artificers", headers=auth_headers())
        assert res.status_code == 200
        assert len(res.json()) == 2

    def test_lists_empty_when_no_artificers(self) -> None:
        res = self.client.get("/api/v1/artificers", headers=auth_headers())
        assert res.status_code == 200
        assert res.json() == []
