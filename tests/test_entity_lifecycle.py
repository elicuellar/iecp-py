import pytest

from iecp_core.types.entity import (
    VALID_LIFECYCLE_TRANSITIONS,
    EntityLifecycleStatus,
    is_valid_lifecycle_transition,
)

# All valid transitions
VALID_TRANSITIONS: list[tuple[EntityLifecycleStatus, EntityLifecycleStatus]] = [
    ("joined", "active"),
    ("joined", "left"),
    ("active", "idle"),
    ("active", "processing"),
    ("active", "disconnected"),
    ("active", "left"),
    ("idle", "active"),
    ("idle", "processing"),
    ("idle", "disconnected"),
    ("idle", "left"),
    ("processing", "active"),
    ("processing", "idle"),
    ("processing", "disconnected"),
    ("processing", "left"),
    ("disconnected", "active"),
    ("disconnected", "idle"),
    ("disconnected", "left"),
]

# All invalid transitions
INVALID_TRANSITIONS: list[tuple[EntityLifecycleStatus, EntityLifecycleStatus]] = [
    ("joined", "idle"),
    ("joined", "processing"),
    ("joined", "disconnected"),
    ("joined", "joined"),
    ("active", "joined"),
    ("active", "active"),
    ("idle", "joined"),
    ("idle", "idle"),
    ("processing", "joined"),
    ("processing", "processing"),
    ("disconnected", "joined"),
    ("disconnected", "processing"),
    ("disconnected", "disconnected"),
    ("left", "joined"),
    ("left", "active"),
]


@pytest.mark.parametrize("from_,to", VALID_TRANSITIONS)
def test_valid_transition(from_: EntityLifecycleStatus, to: EntityLifecycleStatus):
    assert is_valid_lifecycle_transition(from_, to) is True


@pytest.mark.parametrize("from_,to", INVALID_TRANSITIONS)
def test_invalid_transition(from_: EntityLifecycleStatus, to: EntityLifecycleStatus):
    assert is_valid_lifecycle_transition(from_, to) is False


def test_every_status_has_at_least_one_valid_transition():
    all_statuses: list[EntityLifecycleStatus] = [
        "joined", "active", "idle", "processing", "disconnected", "left",
    ]
    for status in all_statuses:
        if status == "left":
            # "left" is terminal — no outgoing transitions, but it IS a valid target
            continue
        assert len(VALID_LIFECYCLE_TRANSITIONS[status]) > 0, (
            f"Status {status} has no valid transitions"
        )


def test_every_target_is_valid_status():
    all_statuses: set[EntityLifecycleStatus] = {
        "joined", "active", "idle", "processing", "disconnected", "left",
    }
    for targets in VALID_LIFECYCLE_TRANSITIONS.values():
        for target in targets:
            assert target in all_statuses, f"Target {target} is not a valid status"
