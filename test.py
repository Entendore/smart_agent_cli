#!/usr/bin/env python3
# test.py
# Test suite for LangGraph-native architecture

import subprocess, sys, os, time, threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Result:
    name: str
    passed: bool
    desc: str
    detail: str = ""


class Tester:
    def __init__(self):
        self.q: Queue = Queue()
        self.lines: List[str] = []
        self.proc: Optional[subprocess.Popen] = None
        self._mark: int = 0

    def _reader(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                if not line:
                    break
                clean = line.rstrip("\n\r")
                self.lines.append(clean)
                self.q.put(clean)
        except Exception:
            pass

    def wait_for(self, pattern: str, timeout: float = 30) -> Tuple[bool, Optional[str]]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self.q.get(timeout=0.5)
                if pattern.lower() in line.lower():
                    return True, line
            except Empty:
                continue
        return False, None

    def wait_prompt(self, timeout: float = 30) -> Tuple[bool, Optional[str]]:
        return self.wait_for("User >>", timeout)

    def send(self, cmd: str, wait: bool = True, timeout: float = 90):
        print(f"\n  [CMD] >>> {cmd if cmd else '(empty)'}", flush=True)
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            return False, str(e)
        if wait:
            return self.wait_prompt(timeout)
        time.sleep(0.3)
        return True, None

    def set_mark(self):
        self._mark = len(self.lines)

    def has(self, pattern: str) -> bool:
        return any(pattern.lower() in l.lower() for l in self.lines)

    def has_after(self, pattern: str) -> bool:
        return any(pattern.lower() in l.lower()
                   for l in self.lines[self._mark:])

    def drain(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Empty:
                break

    def start(self) -> bool:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "main.py"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=0,
                universal_newlines=True, env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
        except Exception as e:
            print(f"  [FATAL] {e}")
            return False
        threading.Thread(target=self._reader, daemon=True).start()
        return self.wait_prompt(60)[0]

    def stop(self):
        if self.proc:
            try:
                self.proc.stdin.write("/exit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


TEST_YAML = """\
defaults:
  model_profile: qwen_cpu

model_profiles:
  qwen_cpu:
    model_name: "qwen2.5:0.5b"
    temperature: 0.0
  qwen_creative:
    model_name: "qwen2.5:0.5b"
    temperature: 0.7

prompt_templates:
  math_specialist: |
    You are a math specialist. Solve precisely. Keep responses very short.
  general_assistant: |
    You are a helpful assistant. Keep responses under 50 words.

workflow:
  pattern: supervisor

agents:
  - name: math_agent
    prompt_blocks:
      - "$math_specialist"
    tools:
      - calculator
    model_profile: qwen_cpu
    metadata:
      role: mathematics
  - name: general_agent
    prompt_blocks:
      - "$general_assistant"
    tools: []
    model_profile: qwen_cpu
    metadata:
      role: general
"""


def check_prereqs() -> bool:
    print("Checking prerequisites...")
    print(f"  Python: {sys.version.split()[0]}")
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True,
                           text=True, timeout=5)
        if r.returncode == 0:
            print(f"  Ollama: {r.stdout.strip()}")
        else:
            print("  Ollama: not running"); return False
    except FileNotFoundError:
        print("  Ollama: not installed"); return False
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        print("  Server: running on :11434")
    except Exception:
        print("  Server: not responding"); return False
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    if "qwen2.5" in r.stdout.lower() or "qwen3" in r.stdout.lower():
        print("  Qwen model: found")
    else:
        print("  Pulling qwen2.5:0.5b ...")
        p = subprocess.run(["ollama", "pull", "qwen2.5:0.5b"],
                           capture_output=True, text=True, timeout=180)
        if p.returncode != 0:
            print(f"  Pull failed: {p.stderr[:80]}"); return False
        print("  Qwen model: pulled")
    for pkg in ["langchain_core", "langchain_openai", "langgraph", "yaml"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  {pkg}: MISSING"); return False
    print("  Packages: OK")
    with open("_test_agents.yaml", "w") as f:
        f.write(TEST_YAML)
    print("  _test_agents.yaml: created")
    print("  All prerequisites met.\n")
    return True


def run_tests() -> List[Result]:
    results: List[Result] = []
    t = Tester()

    def rec(name, passed, desc, detail=""):
        results.append(Result(name, passed, desc, detail))
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}: {desc}")
        if detail and not passed:
            print(f"          -> {detail}")

    # ── 1. Startup ────────────────────────────────────────
    print("\n[1] Startup")
    ok = t.start()
    rec("startup", ok, "CLI starts and shows prompt", "timeout" if not ok else "")
    rec("checkpoint", t.has("MemorySaver"), "MemorySaver enabled")

    # ── 2. Help ───────────────────────────────────────────
    print("\n[2] Help Command")
    t.drain(); t.send("/help", wait=False); time.sleep(0.5)
    for kw in ["Commands", "/load", "/mode", "/profiles", "/list",
               "/resources", "/suggest", "/memory", "/compress",
               "/workflow", "/exit"]:
        rec(f"help_{kw}", t.has(kw), f"shows {kw}")

    # ── 3. Load Config ────────────────────────────────────
    print("\n[3] Load Config")
    t.drain()
    t.send("/load _test_agents.yaml", timeout=30)
    rec("load_ok", t.has("Configuration loaded"), "config loads")
    rec("load_profiles", t.has("Loaded") and t.has("profiles"), "profiles loaded")

    # ── 4. Profiles ──────────────────────────────────────
    print("\n[4] Profiles")
    t.drain(); t.send("/profiles", wait=False); time.sleep(0.5)
    rec("prof_qwen_cpu", t.has("qwen_cpu"), "shows qwen_cpu")
    rec("prof_qwen_creative", t.has("qwen_creative"), "shows qwen_creative")
    rec("prof_marker", t.has("<--"), "marks active")

    # ── 5. List ──────────────────────────────────────────
    print("\n[5] List Agents & Tools")
    t.drain(); t.send("/list", wait=False); time.sleep(0.5)
    rec("list_math", t.has("math_agent"), "shows math_agent")
    rec("list_general", t.has("general_agent"), "shows general_agent")
    rec("list_calc", t.has("calculator"), "shows calculator")
    rec("list_repl", t.has("python_repl"), "shows python_repl")
    rec("list_echo", t.has("echo"), "shows echo")

    # ── 6. Mode Switch ───────────────────────────────────
    print("\n[6] Mode Switch")
    t.drain(); t.send("/mode qwen_creative")
    rec("mode_switch", t.has("Switched to"), "switches profile")
    t.drain(); t.send("/mode qwen_cpu")
    rec("mode_back", t.has("Switched to"), "switches back")

    # ── 7. Invalid Mode ──────────────────────────────────
    print("\n[7] Invalid Mode")
    t.drain(); t.send("/mode nonexistent_xyz")
    rec("invalid_mode", t.has("not found") or t.has("Error"), "shows error")

    # ── 8. Unknown Command ───────────────────────────────
    print("\n[8] Unknown Command")
    t.drain(); t.send("/foobar", wait=False); time.sleep(0.5)
    rec("unknown_cmd", t.has("Unknown"), "shows error")

    # ── 9. Resources ─────────────────────────────────────
    print("\n[9] Resources")
    t.drain(); t.send("/resources", wait=False); time.sleep(0.5)
    rec("res_gpu", t.has("GPU"), "shows GPU")
    rec("res_cpu", t.has("CPU"), "shows CPU")

    # ── 10. Suggest ──────────────────────────────────────
    print("\n[10] Suggest")
    t.drain(); t.send("/suggest", wait=False); time.sleep(0.5)
    rec("suggest_model", t.has("Suggested"), "suggests model")
    rec("suggest_mode", t.has("mode="), "shows mode")

    # ── 11. Workflow ─────────────────────────────────────
    print("\n[11] Workflow")
    t.drain(); t.send("/workflow", wait=False); time.sleep(0.5)
    rec("workflow", t.has("supervisor"), "shows pattern")

    # ── 12. Memory (empty) ───────────────────────────────
    print("\n[12] Memory (empty)")
    t.drain(); t.send("/memory", wait=False); time.sleep(0.5)
    rec("mem_empty", t.has("No messages"), "shows empty")

    # ── 13. Chat — Math ─────────────────────────────────
    print("\n[13] Chat — Math")
    t.drain(); t.set_mark()
    t.send("What is 2+2?", timeout=90)
    rec("math_agent", t.has_after("[math_agent]"), "invokes math_agent")
    rec("math_answer", t.has_after("4") or t.has_after("[Tool]"),
        "has 4 or uses tool")

    # ── 14. Chat — General ──────────────────────────────
    print("\n[14] Chat — General")
    t.drain(); t.set_mark()
    t.send("What color is the sky?", timeout=90)
    rec("gen_agent", t.has_after("[general_agent]"), "invokes general_agent")
    rec("gen_answer", t.has_after("blue"), "mentions blue")

    # ── 15. Chat — Tool Use ─────────────────────────────
    print("\n[15] Chat — Tool Use")
    t.drain(); t.set_mark()
    t.send("Use calculator for 10*5", timeout=90)
    rec("tool_used", t.has_after("[Tool]") or t.has_after("50"),
        "tool runs or shows 50")

    # ── 16. Multi-turn ──────────────────────────────────
    print("\n[16] Multi-turn (MemorySaver persistence)")
    t.drain(); t.set_mark()
    t.send("What is 3+3?", timeout=90)
    time.sleep(1)
    t.send("And 5+5?", timeout=90)
    rec("multi_6", t.has_after("6"), "first answer 6")
    rec("multi_10", t.has_after("10"), "second answer 10")

    # ── 17. Memory (after chat) ─────────────────────────
    print("\n[17] Memory (after chat)")
    t.drain(); t.send("/memory", wait=False); time.sleep(0.5)
    rec("mem_msgs", t.has("Messages:"), "shows count")
    rec("mem_tokens", t.has("Tokens:"), "shows tokens")

    # ── 18. Compress ────────────────────────────────────
    print("\n[18] Compress")
    t.drain(); t.send("/compress", wait=False); time.sleep(3)
    rec("compress", t.has("Compressed") or t.has("new session"),
        "compresses or resets")

    # ── 19. Empty Input ─────────────────────────────────
    print("\n[19] Empty Input")
    t.drain()
    t.send("", wait=False); time.sleep(1)
    ok2 = t.send("/help", timeout=15)[0]
    rec("empty_ok", ok2, "empty input does not crash")

    # ── 20. Exit ────────────────────────────────────────
    print("\n[20] Exit")
    t.send("/exit", wait=False); time.sleep(2)
    rec("exit_clean", t.proc.poll() is not None, "exits cleanly")

    t.stop()
    return results


def summary(results: List[Result]) -> bool:
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    cats: Dict[str, list] = {}
    for r in results:
        if any(x in r.name for x in ["startup", "checkpoint", "help"]):
            c = "UI"
        elif any(x in r.name for x in ["load", "prof"]):
            c = "Config"
        elif any(x in r.name for x in ["mode", "invalid", "unknown"]):
            c = "Error Handling"
        elif any(x in r.name for x in ["list"]):
            c = "Registry"
        elif any(x in r.name for x in ["res", "suggest", "workflow"]):
            c = "Resources"
        elif any(x in r.name for x in ["mem", "compress"]):
            c = "Memory"
        elif any(x in r.name for x in ["math", "gen_", "tool", "multi", "empty"]):
            c = "Chat"
        else:
            c = "Other"
        cats.setdefault(c, [0, 0, []])
        cats[c][0] += int(r.passed)
        cats[c][1] += 1
        cats[c][2].append(r)

    for cat, (p, tot, items) in cats.items():
        icon = "PASS" if p == tot else "FAIL"
        print(f"\n  [{icon}] {cat}: {p}/{tot}")
        for i in items:
            print(f"    {'+' if i.passed else '-'} {i.name}")

    passed = sum(r.passed for r in results)
    total = len(results)
    print(f"\n  TOTAL: {passed}/{total}")
    if passed < total:
        print("\n  FAILED:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}: {r.detail or r.desc}")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Multi-Agent CLI Test Suite")
    print("  LangGraph + MemorySaver + create_react_agent")
    print("=" * 60 + "\n")

    if not check_prereqs():
        print("\nPrerequisites NOT met."); sys.exit(1)

    results = run_tests()
    ok = summary(results)
    try:
        os.remove("_test_agents.yaml")
    except OSError:
        pass
    sys.exit(0 if ok else 1)