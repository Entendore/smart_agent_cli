# registry.py
# Agent and tool storage

from typing import Dict, List, Any
from langchain_core.tools import StructuredTool


class PluginRegistry:
    def __init__(self):
        self.tools: Dict[str, StructuredTool] = {}
        self.agents: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: StructuredTool):
        self.tools[tool.name] = tool

    def register_agent(self, name: str, system_prompt: str,
                       tool_names: List = None, model_profile: str = None,
                       llm_overrides: Dict = None, metadata: Dict = None):
        active = []
        if tool_names:
            for t in tool_names:
                if isinstance(t, dict):
                    if t.get("enabled", True):
                        active.append(t["name"])
                elif isinstance(t, str):
                    active.append(t)
        self.agents[name] = {
            "system_prompt": system_prompt,
            "tool_names": active,
            "model_profile": model_profile,
            "llm_overrides": llm_overrides or {},
            "metadata": metadata or {},
        }

    def get_tools_for_agent(self, agent_name: str) -> List[StructuredTool]:
        cfg = self.agents.get(agent_name, {})
        return [self.tools[t] for t in cfg.get("tool_names", []) if t in self.tools]