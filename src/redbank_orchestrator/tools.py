"""Dynamic routing tools — each tool delegates to a downstream A2A agent.

Tools are created at startup from discovered agent cards, so the orchestrator
automatically adapts when peers are added, removed, or updated.
"""

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from redbank_orchestrator.a2a_client import send_a2a_text_message
from redbank_orchestrator.discovery import PeerAgent

logger = logging.getLogger(__name__)


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


def _build_tool_description(peer: PeerAgent) -> str:
    """Build a tool docstring from the peer's agent card metadata."""
    card = peer.card
    parts = [f"Route a request to the {card.name}."]

    if card.description:
        parts.append(f"\n{card.description}")

    if card.skills:
        parts.append("\n\nCapabilities:")
        for skill in card.skills:
            parts.append(f"  - {skill.name}: {skill.description}")
            if skill.examples:
                examples = ", ".join(f'"{e}"' for e in skill.examples[:3])
                parts.append(f"    Examples: {examples}")

    return "\n".join(parts)


def create_tools_from_peers(peers: list[PeerAgent]) -> list[StructuredTool]:
    """Create LangChain tools dynamically from discovered peer agents.

    Each peer becomes a tool whose name is derived from the agent card name
    and whose description is built from the card's description and skills.
    """
    tools: list[StructuredTool] = []

    for peer in peers:
        # Capture peer in closure
        _peer = peer

        async def _invoke(
            question: str,
            config: RunnableConfig | None = None,
            *,
            _p: PeerAgent = _peer,
        ) -> str:
            auth_token = _get_auth_token(config)
            return await send_a2a_text_message(_p.url, question, auth_token=auth_token)

        tool = StructuredTool.from_function(
            coroutine=_invoke,
            name=peer.tool_name,
            description=_build_tool_description(peer),
            args_schema=AgentQueryInput,
        )
        tools.append(tool)
        logger.info(
            "Created tool %r from peer %s (%s)",
            peer.tool_name,
            peer.card.name,
            peer.url,
        )

    return tools
