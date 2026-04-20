"""
RedBank Orchestrator — A2A server with OpenAI-compatible /chat/completions shim.

Modelled after a2a_langgraph_crewai.langgraph_a2a_server:
  - Starlette app built by A2AStarletteApplication (handles /.well-known/agent-card.json + POST /)
  - Extra routes for /chat/completions, /health, and the playground UI
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from os import getenv
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from redbank_orchestrator.agent import get_graph_closure
from redbank_orchestrator.discovery import PeerAgent
from redbank_orchestrator.tracing import enable_tracing

load_dotenv()

_log_level = getattr(logging, getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(level=_log_level)
logger = logging.getLogger(__name__)

# ── Graph ────────────────────────────────────────────────────────────────────

_graph = None
_peers: list[PeerAgent] = []

# In Docker: main.py + playground/ + images/ are at /opt/app-root/src/
# Locally: they're at the agent root (agents/langgraph/redbank_orchestrator/)
# Either way, main.py imports us, so we resolve relative to main.py's directory.
_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent.parent  # src/../.. = agent root (local)
_CONTAINER_ROOT = Path("/opt/app-root/src")  # container


def _find_path(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # fallback


_PLAYGROUND_HTML = _find_path(
    _CONTAINER_ROOT / "playground" / "templates" / "index.html",
    _PROJECT_ROOT / "playground" / "templates" / "index.html",
)
_IMAGES_DIR = _find_path(
    _CONTAINER_ROOT / "images",
    _PROJECT_ROOT / "images",
)


def _listen_port() -> int:
    return int(getenv("PORT", "8000"))


def _build_graph():
    global _peers
    base_url = getenv("BASE_URL")
    model_id = getenv("MODEL_ID")
    if not base_url or not model_id:
        raise RuntimeError("BASE_URL and MODEL_ID must be set (see .env.example).")
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    graph_closure = get_graph_closure(model_id=model_id, base_url=base_url)
    _peers = getattr(graph_closure, "peers", [])
    return graph_closure()


def _ensure_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


async def run_orchestrator(user_text: str, auth_token: str | None = None) -> str:
    """Shared invoke used by the A2A executor and /chat/completions."""
    graph = _ensure_graph()
    config: dict[str, Any] = {"recursion_limit": 10}
    if auth_token:
        config["configurable"] = {"auth_token": auth_token}
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_text)]}, config=config
    )
    for m in reversed(out.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return ""


# ── A2A executor ─────────────────────────────────────────────────────────────


class OrchestratorA2AExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user = context.get_user_input()
        if not user.strip():
            await event_queue.enqueue_event(
                new_agent_text_message("Error: empty user message.")
            )
            return
        try:
            reply = await run_orchestrator(user)
            await event_queue.enqueue_event(new_agent_text_message(reply))
        except Exception as e:  # noqa: BLE001
            logger.exception("Orchestrator invoke failed")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Orchestrator error: {e!s}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


# ── Starlette route handlers ────────────────────────────────────────────────


def _make_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex[:12]}"


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else str(c)
    return ""


async def _stream_sse(
    user_text: str, model_id: str, auth_token: str | None = None
) -> AsyncIterator[str]:
    """Stream OpenAI chat.completion.chunk SSE events."""
    graph = _ensure_graph()
    completion_id = _make_completion_id()
    created = int(time.time())

    config: dict[str, Any] = {"recursion_limit": 10}
    if auth_token:
        config["configurable"] = {"auth_token": auth_token}

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                text = getattr(chunk, "content", None)
                if text:
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            elif kind == "on_chat_model_end":
                message = event["data"]["output"]
                tool_calls = getattr(message, "tool_calls", None) or []
                if tool_calls:
                    tc_delta = [
                        {
                            "index": i,
                            "id": tc.get("id", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", "")
                                if isinstance(tc, dict)
                                else getattr(tc, "name", ""),
                                "arguments": json.dumps(
                                    tc.get("args", {})
                                    if isinstance(tc, dict)
                                    else getattr(tc, "args", {})
                                ),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ]
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "tool_calls": tc_delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            elif kind == "on_tool_end":
                output = event["data"].get("output", "")
                if hasattr(output, "content"):
                    output = output.content
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "tool",
                                "content": str(output),
                                "name": event.get("name", ""),
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception:
        logger.exception("Stream failed")
        yield f"data: {json.dumps({'error': {'message': 'Internal server error', 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"


async def _chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    stream = bool(body.get("stream", False))
    user_text = _last_user_text(messages)
    if not user_text.strip():
        raise HTTPException(status_code=400, detail="No user message in messages")

    model_id = body.get("model") or getenv("MODEL_ID", "model")
    auth_token = request.headers.get("Authorization")

    if stream:
        return StreamingResponse(
            _stream_sse(user_text, model_id, auth_token),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        reply = await run_orchestrator(user_text, auth_token=auth_token)
    except Exception as e:  # noqa: BLE001
        logger.exception("Chat completions invoke failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(
        {
            "id": _make_completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
        }
    )


async def _health(_request: Request) -> JSONResponse:
    try:
        _ensure_graph()
    except Exception:
        return JSONResponse(
            {"status": "unhealthy", "agent_initialized": False}, status_code=503
        )
    return JSONResponse({"status": "healthy", "agent_initialized": True})


async def _playground_page(_request: Request) -> FileResponse:
    if not _PLAYGROUND_HTML.is_file():
        raise HTTPException(status_code=404, detail="Playground template missing.")
    return FileResponse(_PLAYGROUND_HTML)


async def _serve_image(request: Request) -> FileResponse:
    filename = request.path_params["filename"]
    base = _IMAGES_DIR.resolve()
    file_path = (base / filename).resolve()
    try:
        file_path.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(file_path)


# ── Build app ────────────────────────────────────────────────────────────────


def _build_agent_card() -> AgentCard:
    port = _listen_port()
    public_url = getenv("AGENT_PUBLIC_URL", f"http://localhost:{port}").rstrip("/")

    # Build skills dynamically from discovered peers
    skills: list[AgentSkill] = []
    for peer in _peers:
        card = peer.card
        for skill in card.skills:
            skills.append(
                AgentSkill(
                    id=f"route-{peer.tool_name}-{skill.id}",
                    name=f"{card.name}: {skill.name}",
                    description=skill.description,
                    tags=skill.tags,
                    examples=skill.examples,
                )
            )

    # Fallback if no peers discovered yet (card is built before first request)
    if not skills:
        skills.append(
            AgentSkill(
                id="orchestrator",
                name="Multi-Agent Routing",
                description="Routes user queries to the appropriate specialist agent via A2A discovery.",
                tags=["orchestrator", "routing", "a2a"],
                examples=["How do I reset my password?", "What is my account balance?"],
            )
        )

    peer_names = (
        ", ".join(p.card.name for p in _peers) if _peers else "pending discovery"
    )

    return AgentCard(
        name="RedBank Orchestrator Agent",
        description=(
            f"Multi-agent orchestrator that classifies user intent and routes "
            f"queries to specialist agents via A2A. "
            f"Connected peers: {peer_names}."
        ),
        url=f"{public_url}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
        supports_authenticated_extended_card=False,
    )


def build_app() -> Starlette:
    """Build the Starlette ASGI app with A2A + standard routes."""
    enable_tracing()

    agent_card = _build_agent_card()
    handler = DefaultRequestHandler(
        agent_executor=OrchestratorA2AExecutor(),
        task_store=InMemoryTaskStore(),
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    extra_routes = [
        Route("/", _playground_page, methods=["GET"]),
        Route("/health", _health, methods=["GET"]),
        Route("/chat/completions", _chat_completions, methods=["POST"]),
        Route("/images/{filename:path}", _serve_image, methods=["GET"]),
    ]
    return Starlette(routes=extra_routes + list(a2a_app.routes()))


app = build_app()


def main() -> None:
    port = _listen_port()
    logger.info(
        "RedBank Orchestrator listening on 0.0.0.0:%s; "
        "knowledge_peer=%s banking_peer=%s",
        port,
        getenv("KNOWLEDGE_AGENT_URL", "http://localhost:8001"),
        getenv("BANKING_AGENT_URL", "http://localhost:8002"),
    )
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
