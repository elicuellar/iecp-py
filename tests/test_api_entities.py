"""Entity API tests -- Phase 10."""

from __future__ import annotations

import pytest

from .api_helpers import auth_headers, create_test_app


class TestEntityCreation:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

    def test_creates_entity_and_returns_it(self) -> None:
        res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Alice"},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["entity_type"] == "human"
        assert body["display_name"] == "Alice"
        assert "entity_id" in body

    def test_validates_required_fields(self) -> None:
        res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={},
        )
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_validates_entity_type_enum(self) -> None:
        res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "alien", "display_name": "ET"},
        )
        assert res.status_code == 400

    def test_creates_artificer_entity(self) -> None:
        res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot"},
        )
        assert res.status_code == 201
        assert res.json()["entity_type"] == "artificer"

    def test_creates_daemon_entity(self) -> None:
        res = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "daemon", "display_name": "Daemon"},
        )
        assert res.status_code == 201
        assert res.json()["entity_type"] == "daemon"


class TestGetEntity:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

    def test_retrieves_entity_by_id(self) -> None:
        created = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Bob"},
        ).json()

        res = self.client.get(
            f"/api/v1/entities/{created['entity_id']}",
            headers=auth_headers(),
        )
        assert res.status_code == 200
        assert res.json()["display_name"] == "Bob"

    def test_returns_404_for_nonexistent_entity(self) -> None:
        res = self.client.get("/api/v1/entities/nonexistent", headers=auth_headers())
        assert res.status_code == 404
        assert res.json()["error"]["code"] == "NOT_FOUND"


class TestUpdateEntity:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

    def test_updates_entity_fields(self) -> None:
        created = self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "Bot"},
        ).json()

        res = self.client.patch(
            f"/api/v1/entities/{created['entity_id']}",
            headers=auth_headers(),
            json={"display_name": "SuperBot"},
        )
        assert res.status_code == 200
        assert res.json()["display_name"] == "SuperBot"

    def test_returns_404_for_nonexistent_entity(self) -> None:
        res = self.client.patch(
            "/api/v1/entities/nonexistent",
            headers=auth_headers(),
            json={"display_name": "NewName"},
        )
        assert res.status_code == 404


class TestListEntities:
    def setup_method(self) -> None:
        self.client, self.services = create_test_app()

    def test_lists_entities_filtered_by_type(self) -> None:
        self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "human", "display_name": "Human"},
        )
        self.client.post(
            "/api/v1/entities",
            headers=auth_headers(),
            json={"entity_type": "artificer", "display_name": "AI"},
        )

        res = self.client.get("/api/v1/entities?type=human", headers=auth_headers())
        assert res.status_code == 200
        body = res.json()
        assert len(body) == 1
        assert body[0]["entity_type"] == "human"

    def test_lists_all_entities_without_filter(self) -> None:
        for i in range(3):
            self.client.post(
                "/api/v1/entities",
                headers=auth_headers(),
                json={"entity_type": "human", "display_name": f"User {i}"},
            )

        res = self.client.get("/api/v1/entities", headers=auth_headers())
        assert res.status_code == 200
        assert len(res.json()) == 3

    def test_returns_empty_list_when_no_entities(self) -> None:
        res = self.client.get("/api/v1/entities", headers=auth_headers())
        assert res.status_code == 200
        assert res.json() == []
