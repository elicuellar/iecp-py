"""Lock API tests -- Phase 10."""

from __future__ import annotations

import pytest

from .api_helpers import auth_headers, create_test_app


class TestLockRoutes:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot"},
        )
        self.entity_id = entity_res.json()["entity_id"]

        conv_res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        )
        self.conv_id = conv_res.json()["id"]

    def test_acquires_lock_and_returns_lock_state(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/lock/acquire",
            headers=auth_headers(),
            json={"entityId": self.entity_id},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["granted"] is True
        assert body["lock"] is not None
        assert body["lock"]["holder_id"] == self.entity_id

    def test_denies_when_already_locked_by_another(self) -> None:
        # First acquire
        self.client.post(
            f"/api/v1/conversations/{self.conv_id}/lock/acquire",
            headers=auth_headers(),
            json={"entityId": self.entity_id},
        )

        # Create second entity
        entity2_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "daemon", "display_name": "Bot2"},
        )
        entity2_id = entity2_res.json()["entity_id"]

        # Second acquire -- should be denied
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/lock/acquire",
            headers=auth_headers(),
            json={"entityId": entity2_id},
        )
        assert res.status_code == 200
        assert res.json()["granted"] is False

    def test_releases_held_lock(self) -> None:
        self.client.post(
            f"/api/v1/conversations/{self.conv_id}/lock/acquire",
            headers=auth_headers(),
            json={"entityId": self.entity_id},
        )

        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/lock/release",
            headers=auth_headers(),
            json={"entityId": self.entity_id},
        )
        assert res.status_code == 200
        assert res.json()["released"] is True

    def test_get_lock_state(self) -> None:
        # Initially unlocked
        res1 = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/lock",
            headers=auth_headers(),
        )
        assert res1.status_code == 200
        assert res1.json()["locked"] is False

        # Acquire
        self.client.post(
            f"/api/v1/conversations/{self.conv_id}/lock/acquire",
            headers=auth_headers(),
            json={"entityId": self.entity_id},
        )

        # Now locked
        res2 = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/lock",
            headers=auth_headers(),
        )
        assert res2.status_code == 200
        assert res2.json()["locked"] is True
        assert res2.json()["state"]["holder_id"] == self.entity_id

    def test_acquire_requires_entity_id(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/lock/acquire",
            headers=auth_headers(),
            json={},
        )
        assert res.status_code == 400
