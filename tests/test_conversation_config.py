from iecp_core.types.conversation import (
    ConversationConfig,
    DEFAULT_CONVERSATION_CONFIG,
    validate_conversation_config,
)


def test_default_values():
    c = DEFAULT_CONVERSATION_CONFIG
    assert c.debounce_ms == 3000
    assert c.debounce_adaptive is True
    assert c.lock_ttl_max_ms == 60000
    assert c.allow_human_interrupt is True
    assert c.max_cascade_depth == 3
    assert c.default_respondent_mode == "auto"
    assert c.allow_unsolicited_ai is False
    assert c.context_history_depth == 50
    assert c.context_summary_enabled is True
    assert c.max_participants == 20
    assert c.max_ai_invocations_per_hour == 100
    assert c.max_concurrent_ai_processing == 3
    assert c.decision_capture_enabled is True
    assert c.decision_requires_human_affirmation is True


def test_defaults_pass_validation():
    errors = validate_conversation_config(DEFAULT_CONVERSATION_CONFIG)
    assert errors == []


def test_valid_custom_config():
    config = ConversationConfig(
        debounce_ms=5000,
        lock_ttl_max_ms=120000,
        max_cascade_depth=5,
        context_history_depth=100,
        max_participants=50,
        max_ai_invocations_per_hour=200,
        max_concurrent_ai_processing=5,
    )
    errors = validate_conversation_config(config)
    assert errors == []


def test_negative_debounce_rejected():
    config = ConversationConfig(debounce_ms=-1)
    errors = validate_conversation_config(config)
    assert len(errors) == 1
    assert "debounce_ms" in errors[0]


def test_zero_debounce_accepted():
    config = ConversationConfig(debounce_ms=0)
    errors = validate_conversation_config(config)
    assert errors == []


def test_lock_ttl_below_1000_rejected():
    config = ConversationConfig(lock_ttl_max_ms=999)
    errors = validate_conversation_config(config)
    assert len(errors) == 1
    assert "lock_ttl_max_ms" in errors[0]


def test_negative_cascade_depth_rejected():
    config = ConversationConfig(max_cascade_depth=-1)
    errors = validate_conversation_config(config)
    assert len(errors) == 1
    assert "max_cascade_depth" in errors[0]


def test_zero_cascade_depth_accepted():
    config = ConversationConfig(max_cascade_depth=0)
    errors = validate_conversation_config(config)
    assert errors == []


def test_history_depth_below_1_rejected():
    config = ConversationConfig(context_history_depth=0)
    errors = validate_conversation_config(config)
    assert len(errors) == 1
    assert "context_history_depth" in errors[0]


def test_max_participants_below_2_rejected():
    config = ConversationConfig(max_participants=1)
    errors = validate_conversation_config(config)
    assert len(errors) == 1
    assert "max_participants" in errors[0]


def test_invocations_below_1_rejected():
    config = ConversationConfig(max_ai_invocations_per_hour=0)
    errors = validate_conversation_config(config)
    assert len(errors) == 1
    assert "max_ai_invocations_per_hour" in errors[0]


def test_concurrent_below_1_rejected():
    config = ConversationConfig(max_concurrent_ai_processing=0)
    errors = validate_conversation_config(config)
    assert len(errors) == 1
    assert "max_concurrent_ai_processing" in errors[0]


def test_multiple_errors_collected():
    config = ConversationConfig(
        debounce_ms=-1,
        lock_ttl_max_ms=500,
        max_cascade_depth=-1,
        context_history_depth=0,
        max_participants=1,
        max_ai_invocations_per_hour=0,
        max_concurrent_ai_processing=0,
    )
    errors = validate_conversation_config(config)
    assert len(errors) == 7
