"""LangGraph-backed runner adapter.

The existing AgentRunner contains nanobot-specific context governance, streaming
hooks, tool result persistence, ask-user interruption, and retry recovery.  This
adapter moves the product-layer execution entry point onto a LangGraph graph
while reusing that hardened runner implementation as the first graph node.
"""

from __future__ import annotations

from typing import TypedDict

from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec


class _RunnerState(TypedDict, total=False):
    spec: AgentRunSpec
    result: AgentRunResult


class LangGraphAgentRunner(AgentRunner):
    """Execute AgentRunSpec through a LangGraph graph.

    This is intentionally conservative: the first migration step keeps the
    legacy runner semantics inside a graph node, so existing channels, sessions,
    and tools keep working while the orchestration boundary is now LangGraph.
    """

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        try:
            from langchain_core.runnables import RunnableLambda
            from langgraph.graph import END, START, StateGraph
        except Exception as exc:
            logger.warning("LangGraph unavailable, falling back to AgentRunner: {}", exc)
            return await super().run(spec)

        async def _agent_node(state: _RunnerState) -> _RunnerState:
            result = await AgentRunner.run(self, state["spec"])
            return {"result": result}

        graph = StateGraph(_RunnerState)
        graph.add_node("agent_runner", RunnableLambda(_agent_node))
        graph.add_edge(START, "agent_runner")
        graph.add_edge("agent_runner", END)

        compiled = graph.compile()
        final_state = await compiled.ainvoke({"spec": spec})
        result = final_state.get("result")
        if result is None:
            raise RuntimeError("LangGraph runner finished without an AgentRunResult")
        return result
