"""RedBank Orchestrator Agent — classifies intent and routes to specialist agents via A2A."""

from os import getenv
from typing import Callable

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from redbank_orchestrator.tools import ask_knowledge_agent, ask_banking_agent

SYSTEM_PROMPT = """\
You are the RedBank Orchestrator, a helpful banking assistant that routes user queries \
to the right specialist agent. You have access to two downstream agents:

1. **Knowledge Agent** (ask_knowledge_agent) — for ALL read-only queries:
   - Document and policy questions (password reset, FAQ, bank procedures)
   - Account data retrieval (balance, transaction history, account summary)
   - Any information lookup

2. **Banking Operations Agent** (ask_banking_agent) — for write operations ONLY:
   - Updating account details (address, contact info)
   - Creating transactions (transfers, payments)
   - Any operation that MODIFIES data

ROUTING RULES:
- If the user asks about documents, policies, or procedures -> ask_knowledge_agent
- If the user asks about their account balance, transactions, or account info -> ask_knowledge_agent
- If the user requests an update, change, transfer, or payment -> ask_banking_agent
- If the user greets you or asks a general question you can answer yourself, respond directly
- When in doubt about whether a query is read or write, use ask_knowledge_agent first

IMPORTANT:
- Call exactly ONE tool per user query. Do not call multiple tools for the same question.
- After receiving a tool result, present the answer clearly to the user. Do not call the tool again.
- If a tool returns an access denied error, explain to the user that the operation requires admin privileges.
- Always be professional and concise in your responses.
"""


def get_graph_closure(
    model_id: str = None,
    base_url: str = None,
    api_key: str = None,
) -> Callable:
    """Build and return a closure that creates the RedBank Orchestrator LangGraph agent.

    Returns:
        A function that creates a CompiledGraph agent accepting {"messages": [...]}
        and returns updated state with the routed response.
    """
    if not api_key:
        api_key = getenv("API_KEY")
    if not base_url:
        base_url = getenv("BASE_URL")
    if not model_id:
        model_id = getenv("MODEL_ID")

    is_local = any(host in base_url for host in ["localhost", "127.0.0.1"])

    if not is_local and not api_key:
        raise ValueError("API_KEY is required for non-local environments.")

    tools = [ask_knowledge_agent, ask_banking_agent]

    chat = ChatOpenAI(
        model=model_id,
        temperature=0.01,
        api_key=api_key or "not-needed-for-local-development",
        base_url=base_url,
    )

    def get_graph():
        return create_agent(model=chat, tools=tools, system_prompt=SYSTEM_PROMPT)

    return get_graph
