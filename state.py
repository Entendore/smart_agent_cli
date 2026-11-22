# state.py
# Shared state using LangGraph's add_messages reducer + MemoryManager

from typing import TypedDict, Annotated, List, Dict, Optional, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Graph state. add_messages handles deduplication and append."""
    messages: Annotated[list[BaseMessage], add_messages]


class MemoryManager:
    """Token-aware conversation analysis."""

    CHAR_TO_TOKEN = 0.25
    CONTEXT_WINDOWS: Dict[str, int] = {
        "qwen2.5:0.5b": 32768, "qwen2.5:1.5b": 32768, "qwen2.5:3b": 32768,
        "qwen2.5:7b": 32768, "qwen3:0.6b": 32768, "qwen3:1.7b": 32768,
        "qwen3:4b": 32768, "qwen3:8b": 32768, "llama3:8b": 8192,
        "phi3:mini": 4096, "mistral:7b": 32768,
    }

    def __init__(self, max_messages: int = 50, compression_threshold: int = 25):
        self.max_messages = max_messages
        self.compression_threshold = compression_threshold
        self._summary_llm = None

    def set_summary_llm(self, llm):
        self._summary_llm = llm

    def estimate_tokens(self, messages: List[BaseMessage]) -> int:
        return int(sum(len(str(m.content)) for m in messages) * self.CHAR_TO_TOKEN)

    def trim(self, messages: List[BaseMessage], max_tokens: int = None) -> List[BaseMessage]:
        if not max_tokens:
            return messages[-self.max_messages:]
        result, current = [], 0
        for msg in reversed(messages):
            t = int(len(str(msg.content)) * self.CHAR_TO_TOKEN)
            if current + t > max_tokens:
                break
            result.insert(0, msg)
            current += t
        return result

    def compress(self, messages: List[BaseMessage],
                 llm=None) -> Tuple[List[BaseMessage], bool]:
        if len(messages) <= self.compression_threshold:
            return messages, False
        old = messages[:-self.compression_threshold]
        recent = messages[-self.compression_threshold:]
        summary = self._summarize(old, llm or self._summary_llm)
        if summary:
            return [SystemMessage(content=f"[Previous summary]\n{summary}")] + recent, True
        return messages, False

    def _summarize(self, messages: List[BaseMessage], llm=None) -> Optional[str]:
        convo = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {str(m.content)[:120]}"
            for m in messages)
        if llm:
            try:
                return llm.invoke(
                    f"Summarize in 3-5 bullets:\n{convo}\nSummary:").content
            except Exception:
                pass
        points = [f"- {'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
                  f"{str(m.content)[:80]}" for m in messages[:8]]
        return "Earlier:\n" + "\n".join(points)

    def usage_stats(self, messages: List[BaseMessage], model_name: str) -> Dict:
        total = self.CONTEXT_WINDOWS.get(model_name, 4096)
        used = self.estimate_tokens(messages)
        return {
            "model": model_name, "total_tokens": total,
            "used_tokens": used, "available_tokens": total - used,
            "usage_percent": round((used / total) * 100, 1),
            "message_count": len(messages),
            "warning": (used / total) > 0.8,
        }