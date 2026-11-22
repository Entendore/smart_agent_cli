# main.py
# Entry point — wires MemorySaver into graph and CLI

from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from model_manager import SmartModelManager
from registry import PluginRegistry
from config_loader import PromptManager
from state import MemoryManager
from resource_manager import ResourceManager
from tools import get_builtin_tools
from graph_builder import build_graph
from cli import run_cli


def main():
    # Resource monitor (background, non-blocking)
    rm = ResourceManager(interval=5.0)
    rm.start()

    # Prompt manager with globals
    pm = PromptManager()
    pm.set_global("app_name", "Multi-Agent System")
    pm.set_global("current_date", datetime.now().strftime("%Y-%m-%d"))
    pm.set_global("response_style", "concise")

    # Model manager with resource awareness
    mm = SmartModelManager(resource_manager=rm)

    # Memory manager for token tracking
    mem = MemoryManager(max_messages=50, compression_threshold=25)
    mem.set_summary_llm(mm.get_llm())

    # Plugin registry
    registry = PluginRegistry()
    for tool in get_builtin_tools():
        registry.register_tool(tool)

    # MemorySaver: LangGraph's built-in state persistence
    checkpointer = MemorySaver()

    # Build graph with checkpointer
    app = build_graph(registry, mm, checkpointer)

    # Startup info
    s = rm.status()
    gpu = f"GPU: {s['gpu']['name']}" if s.get("gpu") else "GPU: None"
    print(f"\n[System] {gpu} | CPU RAM: {s['cpu']['available_mb']:.0f}MB free")
    print("[System] Checkpointing: MemorySaver enabled")

    run_cli(registry, mm, app, mem, pm, rm, checkpointer)
    rm.stop()


if __name__ == "__main__":
    main()