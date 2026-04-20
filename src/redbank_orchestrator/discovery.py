"""A2A peer discovery — fetch agent cards from configured peer URLs.

At startup the orchestrator resolves every peer's
``/.well-known/agent-card.json`` and caches the result so tools and the
system prompt can be built dynamically.

Peer URLs come from ``AGENT_URLS`` — a comma-separated list of base URLs.
"""

from __future__ import annotations

import asyncio
import logging
import re
from os import getenv

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCard

logger = logging.getLogger(__name__)


# ── Public helpers ───────────────────────────────────────────────────────────


def get_peer_urls() -> list[str]:
    """Return a deduplicated, ordered list of peer base URLs from env vars."""
    urls: list[str] = []

    raw = getenv("AGENT_URLS", "")
    if raw.strip():
        urls.extend(u.strip().rstrip("/") for u in raw.split(",") if u.strip())

    return urls


def _slugify(name: str) -> str:
    """Turn an agent name like 'Knowledge Agent' into 'ask_knowledge_agent'."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    # Prefix with ask_ for readability unless already present
    if not slug.startswith("ask_"):
        slug = f"ask_{slug}"
    return slug


async def _fetch_card(url: str, timeout: float = 15.0) -> AgentCard | None:
    """Fetch a single peer's agent card. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url=url)
            card = await resolver.get_agent_card()
            logger.info("Discovered peer: %s (%s)", card.name, url)
            return card
    except Exception:
        logger.warning("Failed to discover peer at %s", url, exc_info=True)
        return None


class PeerAgent:
    """Resolved metadata for a single downstream A2A agent."""

    def __init__(self, url: str, card: AgentCard) -> None:
        self.url = url
        self.card = card
        self.tool_name = _slugify(card.name)

    def __repr__(self) -> str:
        return f"PeerAgent(name={self.card.name!r}, url={self.url!r})"


async def discover_peers(
    urls: list[str] | None = None,
    *,
    timeout: float = 15.0,
) -> list[PeerAgent]:
    """Fetch agent cards from all peer URLs concurrently.

    Args:
        urls: Override list of peer base URLs. Defaults to ``get_peer_urls()``.
        timeout: Per-request HTTP timeout.

    Returns:
        A list of successfully discovered ``PeerAgent`` objects.
        Peers that fail to respond are logged and skipped.
    """
    if urls is None:
        urls = get_peer_urls()

    if not urls:
        logger.warning("No peer agent URLs configured — orchestrator has no tools.")
        return []

    cards = await asyncio.gather(*[_fetch_card(u, timeout) for u in urls])

    peers: list[PeerAgent] = []
    seen_names: set[str] = set()
    for url, card in zip(urls, cards):
        if card is None:
            continue
        peer = PeerAgent(url=url, card=card)
        # Ensure unique tool names
        if peer.tool_name in seen_names:
            peer.tool_name = f"{peer.tool_name}_{len(seen_names)}"
        seen_names.add(peer.tool_name)
        peers.append(peer)

    logger.info("Peer discovery complete: %d/%d agents resolved", len(peers), len(urls))
    return peers
