"""Signals, Decisions, and Handoffs API tests -- Phase 10."""

from __future__ import annotations

import pytest

from .api_helpers import auth_headers, create_test_app


class TestSignalRoutes:
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

    def test_emits_signal(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/signals",
            headers=auth_headers(),
            json={"entityId": self.entity_id, "signalType": "listening"},
        )
        assert res.status_code == 201
        assert res.json()["accepted"] is True

    def test_returns_active_signals(self) -> None:
        self.client.post(
            f"/api/v1/conversations/{self.conv_id}/signals",
            headers=auth_headers(),
            json={"entityId": self.entity_id, "signalType": "thinking", "note": "Analyzing..."},
        )

        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/signals",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        body = res.json()
        assert len(body) == 1
        assert body[0]["signal_type"] == "thinking"

    def test_signal_requires_entity_id(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/signals",
            headers=auth_headers(),
            json={"signalType": "thinking"},
        )
        assert res.status_code == 400

    def test_signal_requires_signal_type(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/signals",
            headers=auth_headers(),
            json={"entityId": self.entity_id},
        )
        assert res.status_code == 400


class TestDecisionRoutes:
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

    def test_proposes_a_decision(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/decisions",
            headers=auth_headers(),
            json={"summary": "Use TypeScript for backend", "proposed_by": self.entity_id},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["summary"] == "Use TypeScript for backend"
        assert body["status"] == "proposed"

    def test_affirms_a_decision(self) -> None:
        proposed = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/decisions",
            headers=auth_headers(),
            json={"summary": "Deploy to production", "proposed_by": self.entity_id},
        ).json()

        res = self.client.patch(
            f"/api/v1/conversations/{self.conv_id}/decisions/{proposed['event_id']}",
            headers=auth_headers(),
            json={"action": "affirm", "entity_id": self.entity_id},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "affirmed"

    def test_rejects_a_decision(self) -> None:
        proposed = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/decisions",
            headers=auth_headers(),
            json={"summary": "Use PHP", "proposed_by": self.entity_id},
        ).json()

        res = self.client.patch(
            f"/api/v1/conversations/{self.conv_id}/decisions/{proposed['event_id']}",
            headers=auth_headers(),
            json={"action": "reject", "entity_id": self.entity_id},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "rejected"

    def test_lists_active_decisions(self) -> None:
        self.client.post(
            f"/api/v1/conversations/{self.conv_id}/decisions",
            headers=auth_headers(),
            json={"summary": "Decision 1", "proposed_by": self.entity_id},
        )

        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/decisions",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert len(res.json()) == 1

    def test_returns_404_for_unknown_decision(self) -> None:
        res = self.client.patch(
            f"/api/v1/conversations/{self.conv_id}/decisions/nonexistent",
            headers=auth_headers(),
            json={"action": "affirm", "entity_id": self.entity_id},
        )
        assert res.status_code == 404

    def test_returns_400_for_invalid_action(self) -> None:
        proposed = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/decisions",
            headers=auth_headers(),
            json={"summary": "Some decision", "proposed_by": self.entity_id},
        ).json()

        res = self.client.patch(
            f"/api/v1/conversations/{self.conv_id}/decisions/{proposed['event_id']}",
            headers=auth_headers(),
            json={"action": "invalid_action"},
        )
        assert res.status_code == 400

    def test_supersedes_a_decision(self) -> None:
        proposed = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/decisions",
            headers=auth_headers(),
            json={"summary": "Old decision", "proposed_by": self.entity_id},
        ).json()

        res = self.client.patch(
            f"/api/v1/conversations/{self.conv_id}/decisions/{proposed['event_id']}",
            headers=auth_headers(),
            json={
                "action": "supersede",
                "summary": "New decision",
                "proposed_by": self.entity_id,
            },
        )
        assert res.status_code == 200
        assert res.json()["summary"] == "New decision"


class TestHandoffRoutes:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

        e1_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot1"},
        )
        self.entity1_id = e1_res.json()["entity_id"]

        e2_res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot2"},
        )
        self.entity2_id = e2_res.json()["entity_id"]

        conv_res = self.client.post(
            "/api/v1/conversations",
            headers=auth_headers(),
            json={"created_by": self.entity1_id},
        )
        self.conv_id = conv_res.json()["id"]

        # Create an event to reference
        ev_res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/events",
            headers=auth_headers(),
            json={
                "type": "message",
                "author_id": self.entity1_id,
                "author_type": "artificer",
                "content": {"text": "I need help with this"},
            },
        )
        self.event_id = ev_res.json()["event_id"]

    def test_creates_handoff(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/handoffs",
            headers=auth_headers(),
            json={
                "from_entity": self.entity1_id,
                "to_entity": self.entity2_id,
                "reason": "Needs domain expertise",
                "source_event": self.event_id,
            },
        )
        assert res.status_code == 201
        body = res.json()
        assert body["from_entity"] == self.entity1_id
        assert body["to_entity"] == self.entity2_id

    def test_lists_active_handoffs(self) -> None:
        self.client.post(
            f"/api/v1/conversations/{self.conv_id}/handoffs",
            headers=auth_headers(),
            json={
                "from_entity": self.entity1_id,
                "to_entity": self.entity2_id,
                "reason": "Needs domain expertise",
                "source_event": self.event_id,
            },
        )

        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/handoffs",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert len(res.json()) == 1

    def test_returns_empty_when_no_handoffs(self) -> None:
        res = self.client.get(
            f"/api/v1/conversations/{self.conv_id}/handoffs",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert res.json() == []

    def test_handoff_requires_required_fields(self) -> None:
        res = self.client.post(
            f"/api/v1/conversations/{self.conv_id}/handoffs",
            headers=auth_headers(),
            json={"from_entity": self.entity1_id},
        )
        assert res.status_code == 400
