# cli.py
# CLI using MemorySaver for state persistence — no manual history tracking

import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def print_help():
    print("\n--- Commands ---")
    print("  <msg>        : Chat with agents")
    print("  /load <f>    : Load YAML config")
    print("  /mode <p>    : Switch Model Profile")
    print("  /profiles    : List model profiles")
    print("  /list        : List Agents and Tools")
    print("  /resources   : Show GPU/CPU usage")
    print("  /suggest     : Suggest best model")
    print("  /auto        : Auto-select profile")
    print("  /memory      : Show memory stats")
    print("  /compress    : Compress history")
    print("  /workflow    : Show current workflow")
    print("  /help        : Show this help")
    print("  /exit        : Quit")
    print("----------------")


def run_cli(registry, model_manager, app, memory_manager,
            prompt_manager, resource_manager, checkpointer):

    rm = resource_manager or model_manager.rm
    # MemorySaver uses thread_id for session persistence
    config = {"configurable": {"thread_id": "default-session"}}

    print("\n" + "=" * 60)
    print("  Multi-Agent CLI (LangGraph + MemorySaver)")
    print("=" * 60)
    print("System Ready. Type '/load agents.yaml' to start.")
    print_help()

    while True:
        try:
            user_input = input("\nUser >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""

            if cmd == "/exit":
                print("Goodbye!")
                break
            elif cmd == "/help":
                print_help()

            elif cmd == "/load" and arg:
                from config_loader import ConfigLoader
                if ConfigLoader.load(arg, registry, model_manager, prompt_manager):
                    from graph_builder import build_graph
                    app = build_graph(registry, model_manager, checkpointer)
                    # New config = new session
                    config["configurable"]["thread_id"] = f"session-{time.time()}"

            elif cmd == "/mode" and arg:
                if model_manager.set_mode(arg):
                    from graph_builder import build_graph
                    app = build_graph(registry, model_manager, checkpointer)
                    config["configurable"]["thread_id"] = f"session-{time.time()}"

            elif cmd == "/profiles":
                print("\n--- Profiles ---")
                for k, v in model_manager.profiles.items():
                    marker = " <--" if k == model_manager.current_profile_name else ""
                    print(f"  {k}: {v.get('model_name')} temp={v.get('temperature', 0)}{marker}")

            elif cmd == "/list":
                print("\n--- Agents ---")
                if not registry.agents:
                    print("  (none)")
                for k, v in registry.agents.items():
                    print(f"  {k}: tools={v['tool_names']} meta={v.get('metadata')}")
                print("--- Tools ---")
                for k, v in registry.tools.items():
                    print(f"  {k}: {v.description[:60] if v.description else ''}")

            elif cmd == "/resources":
                s = rm.status()
                print(f"\n--- Resources ---")
                print(f"  GPU: {'Yes' if s['has_gpu'] else 'No'}")
                if s.get("gpu"):
                    g = s["gpu"]
                    print(f"  {g['name']}: {g['mem_used_mb']}/{g['mem_total_mb']}MB "
                          f"({g['available_mb']}MB free)")
                c = s["cpu"]
                print(f"  CPU RAM: {c['mem_used_mb']}/{c['mem_total_mb']}MB "
                      f"({c['available_mb']}MB free)")

            elif cmd == "/suggest":
                suggested = rm.suggest_model()
                check = rm.can_load(suggested)
                print(f"\n  Suggested: {suggested} (mode={check['mode']})")

            elif cmd == "/auto":
                selected = model_manager.auto_select()
                print(f"\n  Auto-selected: {selected}")
                from graph_builder import build_graph
                app = build_graph(registry, model_manager, checkpointer)
                config["configurable"]["thread_id"] = f"session-{time.time()}"

            elif cmd == "/memory":
                try:
                    state = app.get_state(config)
                    msgs = state.values.get("messages", []) if state.values else []
                except Exception:
                    msgs = []
                if msgs:
                    stats = memory_manager.usage_stats(msgs, model_manager.current_model_name)
                    print(f"\n--- Memory ---")
                    print(f"  Messages: {stats['message_count']}")
                    print(f"  Tokens: {stats['used_tokens']}/{stats['total_tokens']} "
                          f"({stats['usage_percent']}%)")
                    if stats["warning"]:
                        print("  Warning: High usage — /compress")
                else:
                    print("\n  No messages in history.")

            elif cmd == "/compress":
                try:
                    state = app.get_state(config)
                    msgs = state.values.get("messages", []) if state.values else []
                except Exception:
                    msgs = []
                if msgs:
                    old = len(msgs)
                    _, ok = memory_manager.compress(msgs, model_manager.get_llm())
                    if ok:
                        # Start fresh session (compressed context lost on rebuild)
                        config["configurable"]["thread_id"] = f"compressed-{time.time()}"
                        print(f"\n  Compressed: {old} messages (new session)")
                    else:
                        print("\n  Nothing to compress.")
                else:
                    print("\n  Nothing to compress.")

            elif cmd == "/workflow":
                wcfg = getattr(registry, "_workflow_cfg", None)
                if isinstance(wcfg, dict):
                    print(f"\n  Pattern: {wcfg.get('pattern', 'supervisor')}")
                else:
                    print("\n  Pattern: supervisor (default)")

            else:
                print(f"  Unknown: {cmd}. Type /help")
            continue

        # ── Chat: stream via MemorySaver ────────────────
        # MemorySaver auto-loads previous messages from checkpoint.
        # We only send the new HumanMessage.
        try:
            for event in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="updates"
            ):
                for node, update in event.items():
                    if node == "supervisor":
                        continue
                    if "messages" not in update:
                        continue
                    for msg in update["messages"]:
                        if isinstance(msg, ToolMessage):
                            content = str(msg.content)
                            print(f"\n[Tool] {content[:300]}")
                        elif isinstance(msg, AIMessage) and msg.content:
                            print(f"\n[{node}]")
                            print(f"  {msg.content}")
        except Exception as e:
            print(f"\n[Error] {e}")