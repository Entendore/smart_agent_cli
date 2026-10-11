# graph_builder.py
# LangGraph best practices: create_react_agent, Command routing, MemorySaver

from enum import Enum
from typing import Dict, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from state import AgentState


class Pattern(Enum):
    SUPERVISOR = "supervisor"
    SWARM = "swarm"
    PIPELINE = "pipeline"


def build_graph(registry, model_manager, checkpointer=None, pattern: str = None):
    """Build workflow using LangGraph best practices."""
    agents = list(registry.agents.keys())
    if not agents:
        return _empty_graph(checkpointer)

    cfg = getattr(registry, "_workflow_cfg", {})
    p = pattern or (cfg.get("pattern", "supervisor") if isinstance(cfg, dict) else "supervisor")
    try:
        pat = Pattern(p)
    except ValueError:
        pat = Pattern.SUPERVISOR

    builders = {
        Pattern.SUPERVISOR: _supervisor,
        Pattern.SWARM: _swarm,
        Pattern.PIPELINE: _pipeline,
    }
    return builders[pat](registry, model_manager, agents,
                         cfg if isinstance(cfg, dict) else {}, checkpointer)


def _empty_graph(checkpointer=None):
    w = StateGraph(AgentState)
    w.add_node("idle", lambda s: {"messages": []})
    w.set_entry_point("idle")
    w.add_edge("idle", END)
    return w.compile(checkpointer=checkpointer)


def _make_worker(registry, name, model_manager):
    """Build a worker using create_react_agent (handles tool loops automatically)."""
    cfg = registry.agents[name]
    tools = registry.get_tools_for_agent(name)
    llm = model_manager.get_llm(
        override_profile=cfg.get("model_profile"),
        llm_overrides=cfg.get("llm_overrides"))

    try:
        # create_react_agent: builds a subgraph with agent + tool loop
        agent = create_react_agent(llm, tools, prompt=cfg["system_prompt"])
    except TypeError:
        # Fallback for older langgraph without prompt param
        agent = _manual_react(llm, tools, cfg["system_prompt"])

    return agent


def _manual_react(llm, tools, system_prompt):
    """Fallback: build react loop manually using LangGraph primitives."""
    from langgraph.graph import StateGraph, END
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    w = StateGraph(AgentState)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | llm.bind_tools(tools) if tools else prompt | llm

    def agent_fn(state):
        result = chain.invoke({"messages": state["messages"]})
        return {"messages": [result]}

    w.add_node("agent", agent_fn)
    if tools:
        w.add_node("tools", ToolNode(tools))
        w.add_conditional_edges(
            "agent",
            lambda s: "tools" if (s["messages"] and
                hasattr(s["messages"][-1], "tool_calls") and
                s["messages"][-1].tool_calls) else END,
            {"tools": "tools", END: END})
        w.add_edge("tools", "agent")
    else:
        w.add_edge("agent", END)
    w.set_entry_point("agent")
    return w.compile()


# ─── SUPERVISOR with Command routing ──────────────────────
def _supervisor(registry, mm, agents, cfg, checkpointer):
    workflow = StateGraph(AgentState)

    # Worker subgraphs (each handles its own tool loop)
    for name in agents:
        worker = _make_worker(registry, name, mm)
        workflow.add_node(name, worker)

    # Supervisor node
    sup_llm = mm.get_llm()
    members = ", ".join(agents)
    sup_prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"Pick one worker. Reply with ONLY the name.\n"
         f"Workers:\n" + "\n".join(f"- {a}" for a in agents) +
         "\nIf done: FINISH"),
        MessagesPlaceholder("messages"),
    ])

    def supervisor(state: AgentState):
        result = (sup_prompt | sup_llm).invoke({"messages": state["messages"]})
        content = result.content.strip().lower()
        # Use Command for explicit routing — overrides default edge
        for a in agents:
            if a.replace("_", "") in content.replace("_", "").replace(" ", ""):
                return Command(goto=a)
        # No Command returned → default edge to END applies
        return {}

    workflow.add_node("supervisor", supervisor)

    # Default edge: supervisor → END (used when no Command returned)
    workflow.add_edge("supervisor", END)

    # Workers always return to supervisor after finishing
    for a in agents:
        workflow.add_edge(a, "supervisor")

    workflow.set_entry_point("supervisor")
    return workflow.compile(checkpointer=checkpointer)


# ─── SWARM: peer-to-peer with Command ─────────────────────
def _swarm(registry, mm, agents, cfg, checkpointer):
    workflow = StateGraph(AgentState)

    for name in agents:
        worker = _make_worker(registry, name, mm)
        workflow.add_node(name, worker)

    def first_router(state: AgentState):
        return Command(goto=agents[0])

    def make_swarm_edge(current):
        def route(state: AgentState):
            msgs = state.get("messages", [])
            if msgs:
                c = str(msgs[-1].content).lower()
                for a in agents:
                    if a.replace("_", "") in c.replace("_", "").replace(" ", ""):
                        if a != current:
                            return Command(goto=a)
            return Command(goto="__end__")
        return route

    workflow.add_node("router", first_router)
    workflow.add_edge("router", agents[0])

    for i, a in enumerate(agents):
        next_options = [x for x in agents if x != a] + [END]
        workflow.add_conditional_edges(a, make_swarm_edge(a),
                                      {x: x for x in next_options})

    workflow.set_entry_point("router")
    return workflow.compile(checkpointer=checkpointer)


# ─── PIPELINE: sequential stages ──────────────────────────
def _pipeline(registry, mm, agents, cfg, checkpointer):
    workflow = StateGraph(AgentState)
    stages = cfg.get("stages", agents)
    valid = [s for s in stages if s in registry.agents] or agents[:1]

    for name in valid:
        worker = _make_worker(registry, name, mm)
        workflow.add_node(name, worker)

    for i in range(len(valid) - 1):
        workflow.add_edge(valid[i], valid[i + 1])
    workflow.add_edge(valid[-1], END)
    workflow.set_entry_point(valid[0])
    return workflow.compile(checkpointer=checkpointer)