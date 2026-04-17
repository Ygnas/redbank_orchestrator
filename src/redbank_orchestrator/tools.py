"""Routing tools — each tool delegates to a downstream A2A agent.

The orchestrator LLM decides which tool to call based on user intent.
Auth tokens are propagated from the incoming request via LangGraph config.
"""

from os import getenv

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from redbank_orchestrator.a2a_client import send_a2a_text_message


class AgentQueryInput(BaseModel):
    """Input schema for routing a user query to a downstream agent."""

    question: str = Field(
        description="The user's question or request to forward to the agent."
    )


def _get_auth_token(config: RunnableConfig | None) -> str | None:
    """Extract the auth token from the LangGraph RunnableConfig."""
    if config and "configurable" in config:
        return config["configurable"].get("auth_token")
    return None


@tool("ask_knowledge_agent", parse_docstring=True)
async def ask_knowledge_agent(
    question: str, config: RunnableConfig | None = None
) -> str:
    """Route a question to the Knowledge Agent for document or account data retrieval.

    Use this tool for questions about bank documents, policies, procedures,
    account balances, transaction history, and any read-only information retrieval.
    The Knowledge Agent has access to a PGVector knowledge base and a PostgreSQL
    MCP server with customer data (read-only, RLS-scoped per user).

    Args:
        question: The user's question about documents, policies, or account data.

    Returns:
        The Knowledge Agent's response with the requested information.
    """
    url = getenv("KNOWLEDGE_AGENT_URL", "http://localhost:8001")
    auth_token = _get_auth_token(config)
    return await send_a2a_text_message(url, question, auth_token=auth_token)


@tool("ask_banking_agent", parse_docstring=True)
async def ask_banking_agent(question: str, config: RunnableConfig | None = None) -> str:
    """Route a request to the Banking Operations Agent for write operations.

    Use this tool ONLY for updating account details, creating new transactions,
    or any operation that modifies customer or account data. This agent is
    restricted to admin users only. Non-admin users will receive an access
    denied response. Do NOT use this for read-only queries.

    Args:
        question: The user's request for a banking write operation.

    Returns:
        The Banking Agent's response confirming the operation or access denial.
    """
    url = getenv("BANKING_AGENT_URL", "http://localhost:8002")
    auth_token = _get_auth_token(config)
    return await send_a2a_text_message(url, question, auth_token=auth_token)
