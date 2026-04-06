"""Conversation API tests -- Phase 10."""

from __future__ import annotations

import pytest

from .api_helpers import auth_headers, create_test_app


class TestConversationCreation:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()
        # Create entity to own conversations
        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Owner"},
        )
        self.entity_id = entity_res.json()["entity_id"]

    def test_creates_conversation(self) -> None:
        res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id, "title": "Test Conv"},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["title"] == "Test Conv"
        assert "id" in body

    def test_validates_required_fields(self) -> None:
        res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={},
        )
        assert res.status_code == 400

    def test_creates_conversation_without_title(self) -> None:
        res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        )
        assert res.status_code == 201


class TestGetConversation:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()
        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Owner"},
        )
        self.entity_id = entity_res.json()["entity_id"]

    def test_retrieves_conversation_with_participants(self) -> None:
        created = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        ).json()

        res = self.client.get(
            f"/api/v1/conversations/{created['id']}",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        body = res.json()
        assert len(body["participants"]) == 1  # creator auto-added
        assert body["participants"][0]["entity_id"] == self.entity_id

    def test_returns_404_for_nonexistent_conversation(self) -> None:
        res = self.client.get("/api/v1/conversations/nonexistent", headers=auth_headers())
        assert res.status_code == 404


class TestUpdateConversation:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()
        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Owner"},
        )
        self.entity_id = entity_res.json()["entity_id"]

    def test_updates_conversation_config(self) -> None:
        conv = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        ).json()

        res = self.client.patch(
            f"/api/v1/conversations/{conv['id']}",
            headers=auth_headers(),
            json={"config": {"debounce_ms": 5000}},
        )
        assert res.status_code == 200
        assert res.json()["config"]["debounce_ms"] == 5000


class TestParticipantManagement:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Owner"},
        )
        self.entity_id = entity_res.json()["entity_id"]

        conv_res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        )
        self.conv_id = conv_res.json()["id"]

    def test_adds_participant(self) -> None:
        entity2_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot"},
        )
        entity2_id = entity2_res.json()["entity_id"]

        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/participants",
            headers=auth_headers(),
            json={"entity_id": entity2_id},
        )
        assert res.status_code == 201
        assert res.json()["entity_id"] == entity2_id
        assert res.json()["role"] == "member"

    def test_removes_participant(self) -> None:
        entity2_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot"},
        )
        entity2_id = entity2_res.json()["entity_id"]

        self.client.post(
            f"/api/v1/conversations/{self.conv_id}/participants",
            headers=auth_headers(),
            json={"entity_id": entity2_id},
        )

        res = self.client.delete(
            f"/api/v1/conversations/{self.conv_id}/participants/{entity2_id}",
            headers=auth_headers(),
        )
        assert res.status_code == 204

        # Verify participant is gone
        participants_res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/participants",
            headers=auth_headers(),
        )
        participant_ids = [p["entity_id"] for p in participants_res.json()]
        assert entity2_id not in participant_ids

    def test_rejects_duplicate_participant(self) -> None:
        # Creator is already a participant
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/participants",
            headers=auth_headers(),
            json={"entity_id": self.entity_id},
        )
        assert res.status_code == 409

    def test_lists_participants(self) -> None:
        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/participants",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert len(res.json()) == 1  # Just the creator
