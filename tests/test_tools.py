import sys
import os
from unittest.mock import AsyncMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.redbank_orchestrator.tools import (
    ask_knowledge_agent,
    ask_banking_agent,
    AgentQueryInput,
    _get_auth_token,
)


# ── Schema tests ─────────────────────────────────────────────────────────────


def test_agent_query_input_schema():
    """Test that the AgentQueryInput schema is properly defined."""
    schema = AgentQueryInput(question="What is my account balance?")
    assert schema.question == "What is my account balance?"


def test_agent_query_input_has_description():
    """Test that schema fields have descriptions."""
    json_schema = AgentQueryInput.model_json_schema()
    assert "properties" in json_schema
    assert "question" in json_schema["properties"]
    assert "description" in json_schema["properties"]["question"]


# ── Tool definition tests ────────────────────────────────────────────────────


def test_knowledge_agent_tool_exists():
    """Test that ask_knowledge_agent tool is properly defined."""
    assert ask_knowledge_agent is not None
    assert ask_knowledge_agent.name == "ask_knowledge_agent"
    assert ask_knowledge_agent.description is not None


def test_banking_agent_tool_exists():
    """Test that ask_banking_agent tool is properly defined."""
    assert ask_banking_agent is not None
    assert ask_banking_agent.name == "ask_banking_agent"
    assert ask_banking_agent.description is not None


def test_knowledge_agent_tool_description_content():
    """Test that knowledge agent tool description covers expected use cases."""
    desc = ask_knowledge_agent.description.lower()
    assert "knowledge" in desc or "document" in desc or "read" in desc


def test_banking_agent_tool_description_content():
    """Test that banking agent tool description covers expected use cases."""
    desc = ask_banking_agent.description.lower()
    assert "banking" in desc or "write" in desc or "operation" in desc


def test_tools_have_args_schema():
    """Test that both tools have properly configured args schemas."""
    assert hasattr(ask_knowledge_agent, "args_schema")
    assert hasattr(ask_banking_agent, "args_schema")


# ── Auth token extraction tests ──────────────────────────────────────────────


def test_get_auth_token_from_config():
    """Test that auth token is extracted from RunnableConfig."""
    config = {"configurable": {"auth_token": "Bearer test-token-123"}}
    assert _get_auth_token(config) == "Bearer test-token-123"


def test_get_auth_token_missing_config():
    """Test that None is returned when config is None."""
    assert _get_auth_token(None) is None


def test_get_auth_token_missing_configurable():
    """Test that None is returned when configurable is missing."""
    assert _get_auth_token({}) is None


def test_get_auth_token_missing_auth_token():
    """Test that None is returned when auth_token is not in configurable."""
    assert _get_auth_token({"configurable": {}}) is None


# ── Tool invocation tests (mocked A2A client) ────────────────────────────────


@pytest.mark.asyncio
@patch("src.redbank_orchestrator.tools.send_a2a_text_message", new_callable=AsyncMock)
async def test_knowledge_agent_invokes_a2a(mock_send):
    """Test that ask_knowledge_agent calls the A2A client with correct URL."""
    mock_send.return_value = "Your account balance is $1,234.56"

    with patch.dict(os.environ, {"KNOWLEDGE_AGENT_URL": "http://knowledge:8080"}):
        result = await ask_knowledge_agent.ainvoke({"question": "What is my balance?"})

    assert result == "Your account balance is $1,234.56"
    mock_send.assert_called_once_with(
        "http://knowledge:8080",
        "What is my balance?",
        auth_token=None,
    )


@pytest.mark.asyncio
@patch("src.redbank_orchestrator.tools.send_a2a_text_message", new_callable=AsyncMock)
async def test_banking_agent_invokes_a2a(mock_send):
    """Test that ask_banking_agent calls the A2A client with correct URL."""
    mock_send.return_value = "Transaction created successfully."

    with patch.dict(os.environ, {"BANKING_AGENT_URL": "http://banking:8080"}):
        result = await ask_banking_agent.ainvoke(
            {"question": "Transfer $100 to account 123"}
        )

    assert result == "Transaction created successfully."
    mock_send.assert_called_once_with(
        "http://banking:8080",
        "Transfer $100 to account 123",
        auth_token=None,
    )


@pytest.mark.asyncio
@patch("src.redbank_orchestrator.tools.send_a2a_text_message", new_callable=AsyncMock)
async def test_knowledge_agent_default_url(mock_send):
    """Test that ask_knowledge_agent uses default URL when env var is not set."""
    mock_send.return_value = "Response"

    # Clear env var to test default
    env = os.environ.copy()
    env.pop("KNOWLEDGE_AGENT_URL", None)
    with patch.dict(os.environ, env, clear=True):
        await ask_knowledge_agent.ainvoke({"question": "test"})

    # Should use default localhost:8001
    call_args = mock_send.call_args
    assert "localhost:8001" in call_args[0][0]


@pytest.mark.asyncio
@patch("src.redbank_orchestrator.tools.send_a2a_text_message", new_callable=AsyncMock)
async def test_banking_agent_default_url(mock_send):
    """Test that ask_banking_agent uses default URL when env var is not set."""
    mock_send.return_value = "Response"

    env = os.environ.copy()
    env.pop("BANKING_AGENT_URL", None)
    with patch.dict(os.environ, env, clear=True):
        await ask_banking_agent.ainvoke({"question": "test"})

    call_args = mock_send.call_args
    assert "localhost:8002" in call_args[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
