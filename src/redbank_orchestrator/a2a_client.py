"""Call another A2A agent and turn SendMessageResponse into plain text.

Based on a2a_langgraph_crewai.a2a_reply with added auth token forwarding.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    SendMessageRequest,
    Task,
)
from a2a.utils import get_artifact_text, get_message_text

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*A2AClient is deprecated.*",
    category=DeprecationWarning,
)


def _unwrap_send_result(response: Any) -> Any:
    root = response.root
    if isinstance(root, JSONRPCErrorResponse):
        raise RuntimeError(f"A2A JSON-RPC error: {root.error}")
    return root.result


def _result_to_text(result: Message | Task | Any) -> str:
    if isinstance(result, Message):
        return get_message_text(result)
    if isinstance(result, Task):
        chunks: list[str] = []
        if result.artifacts:
            for art in result.artifacts:
                chunks.append(get_artifact_text(art))
        if chunks:
            return "\n".join(chunks)
        if result.status and result.status.message:
            return get_message_text(result.status.message)
        return str(result)
    return str(result)


async def send_a2a_text_message(
    base_url: str,
    text: str,
    auth_token: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Fetch agent card, send one user text message, return assistant text.

    Args:
        base_url: Base URL of the downstream A2A agent.
        text: The user's message text to forward.
        auth_token: Optional Bearer token for AuthBridge identity propagation.
        timeout: HTTP timeout in seconds.
    """
    base = base_url.rstrip("/")

    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = (
            auth_token
            if auth_token.lower().startswith("bearer ")
            else f"Bearer {auth_token}"
        )

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        resolver = A2ACardResolver(httpx_client=client, base_url=base)
        card = await resolver.get_agent_card()
        a2a = A2AClient(httpx_client=client, agent_card=card)

        payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "messageId": uuid4().hex,
            },
        }
        req = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload),
        )

        logger.info("A2A -> peer=%s id=%s text_len=%d", base, req.id, len(text))

        try:
            resp = await a2a.send_message(req)
        except Exception:
            logger.exception("A2A <- peer=%s failed (id=%s)", base, req.id)
            raise

        result = _unwrap_send_result(resp)
        out = _result_to_text(result)
        logger.info("A2A <- peer=%s id=%s ok result_len=%d", base, req.id, len(out))
        return out
