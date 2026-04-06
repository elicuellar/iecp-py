"""Event API tests -- Phase 10."""

from __future__ import annotations

import pytest

from .api_helpers import auth_headers, create_test_app


class TestEventAppend:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Alice"},
        )
        self.entity_id = entity_res.json()["entity_id"]

        conv_res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        )
        self.conv_id = conv_res.json()["id"]

    def test_appends_message_event(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "message",
                "author_id": self.entity_id,
                "author_type": "human",
                "content": {"text": "Hello, world!", "format": "plain", "mentions": []},
            },
        )
        assert res.status_code == 201
        body = res.json()
        assert body["event_type"] == "message"
        assert body["content"]["text"] == "Hello, world!"
        assert "event_id" in body

    def test_validates_required_fields(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={"type": "message"},
        )
        assert res.status_code == 400

    def test_rejects_unknown_event_type(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "unknown_type",
                "author_id": self.entity_id,
                "author_type": "human",
                "content": {},
            },
        )
        assert res.status_code == 400

    def test_appends_action_event(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "action",
                "author_id": self.entity_id,
                "author_type": "human",
                "content": {"action_type": "search", "description": "Searching for data"},
            },
        )
        assert res.status_code == 201
        assert res.json()["event_type"] == "action"

    def test_appends_system_event(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "system",
                "author_id": self.entity_id,
                "author_type": "system",
                "content": {"system_event": "connected", "description": "User connected"},
            },
        )
        assert res.status_code == 201
        assert res.json()["event_type"] == "system"


class TestReadEvents:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Alice"},
        )
        self.entity_id = entity_res.json()["entity_id"]

        conv_res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        )
        self.conv_id = conv_res.json()["id"]

    def test_reads_events_with_pagination(self) -> None:
        ev1 = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "message",
                "author_id": self.entity_id,
                "author_type": "human",
                "content": {"text": "First"},
            },
        ).json()

        self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "message",
                "author_id": self.entity_id,
                "author_type": "human",
                "content": {"text": "Second"},
            },
        )

        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/events?after={ev1['event_id']}",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        body = res.json()
        assert len(body["events"]) == 1
        assert body["events"][0]["content"]["text"] == "Second"

    def test_reads_all_events_without_cursor(self) -> None:
        for i in range(3):
            self.client.post(
                f"/api/v1/conversations/{self.conv_id}/events",
                headers=auth_headers(),
                json={
                    "type": "message",
                    "author_id": self.entity_id,
                    "author_type": "human",
                    "content": {"text": f"Message {i}"},
                },
            )

        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert len(res.json()["events"]) == 3


class TestGetSingleEvent:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Alice"},
        )
        self.entity_id = entity_res.json()["entity_id"]

        conv_res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        )
        self.conv_id = conv_res.json()["id"]

    def test_gets_single_event(self) -> None:
        created = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "message",
                "author_id": self.entity_id,
                "author_type": "human",
                "content": {"text": "Hello"},
            },
        ).json()

        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/events/{created['event_id']}",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert res.json()["content"]["text"] == "Hello"

    def test_returns_404_for_nonexistent_event(self) -> None:
        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/events/nonexistent",
            headers=auth_headers(),
        )
        assert res.status_code == 404


class TestEditEvent:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Alice"},
        )
        self.entity_id = entity_res.json()["entity_id"]

        conv_res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        )
        self.conv_id = conv_res.json()["id"]

    def test_edits_event_sets_status_edited(self) -> None:
        created = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "message",
                "author_id": self.entity_id,
                "author_type": "human",
                "content": {"text": "Original"},
            },
        ).json()

        res = self.client.patch(
            f"/api/v1/conversations/{self.conv_id}/events/{created['event_id']}",
            headers=auth_headers(),
            json={"content": {"text": "Edited"}},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "edited"


class TestDeleteEvent:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        entity_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Alice"},
        )
        self.entity_id = entity_res.json()["entity_id"]

        conv_res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity_id},
        )
        self.conv_id = conv_res.json()["id"]

    def test_soft_deletes_event(self) -> None:
        created = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "message",
                "author_id": self.entity_id,
                "author_type": "human",
                "content": {"text": "Delete me"},
            },
        ).json()

        res = self.client.delete(
            f"/api/v1/conversations/{self.conv_id}/events/{created['event_id']}",
            headers=auth_headers(),
        )
        assert res.status_code == 204

        # Verify status changed
        check = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/events/{created['event_id']}",
            headers=auth_headers(),
        )
        assert check.json()["status"] == "deleted"
