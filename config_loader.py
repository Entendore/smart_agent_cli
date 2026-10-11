# config_loader.py
# YAML parsing with per-section error handling, prompt templates with variables

import yaml, re
from typing import Dict, List, Optional, Any
from langchain_core.tools import StructuredTool


class PromptTemplate:
    def __init__(self, name: str, content: str):
        self.name = name
        self.variables = re.findall(r'\{\{(\w+)\}\}', content)
        self.versions: Dict[str, str] = {"1.0": content}
        self.current = "1.0"

    def add_version(self, version: str, content: str):
        self.versions[version] = content

    def set_version(self, v: str):
        if v in self.versions:
            self.current = v

    def render(self, variables: Dict[str, Any] = None, version: str = None) -> str:
        content = self.versions.get(version or self.current, "")
        if variables:
            for k, v in variables.items():
                content = content.replace(f"{{{{{k}}}}}", str(v))
        return content


class PromptManager:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.globals: Dict[str, Any] = {}

    def register(self, name: str, content: str) -> PromptTemplate:
        pt = PromptTemplate(name, content)
        self.templates[name] = pt
        return pt

    def render(self, name: str, variables: Dict = None, version: str = None) -> str:
        pt = self.templates.get(name)
        if not pt:
            return f"[Missing: {name}]"
        return pt.render({**self.globals, **(variables or {})}, version)

    def set_global(self, key: str, value: Any):
        self.globals[key] = value


class ConfigLoader:
    @staticmethod
    def load(filepath: str, registry, model_manager,
             prompt_manager: PromptManager = None) -> bool:
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"[Error] Cannot read {filepath}: {e}")
            return False

        errors: List[str] = []

        if "model_profiles" in data:
            try:
                defaults = data.get("defaults", {})
                model_manager.load_profiles(
                    data["model_profiles"],
                    defaults.get("model_profile", "default"))
            except Exception as e:
                errors.append(f"Profiles: {e}")

        tools_list = data.get("tools", data.get("custom_tools", []))
        for t_conf in tools_list:
            try:
                tool = ConfigLoader._make_tool(t_conf)
                if tool:
                    registry.register_tool(tool)
            except Exception as e:
                errors.append(f"Tool '{t_conf.get('name', '?')}': {e}")

        pm = prompt_manager
        if pm and "prompt_templates" in data:
            try:
                for name, val in data["prompt_templates"].items():
                    try:
                        if isinstance(val, str):
                            pm.register(name, val)
                        elif isinstance(val, dict):
                            pt = pm.register(name, val["content"])
                            for ver, vd in val.get("versions", {}).items():
                                pt.add_version(ver, vd["content"])
                            if "default_version" in val:
                                pt.set_version(val["default_version"])
                    except Exception as e:
                        errors.append(f"Template '{name}': {e}")
            except Exception as e:
                errors.append(f"Templates: {e}")

        if pm and "prompt_globals" in data:
            try:
                pm.globals.update(data["prompt_globals"])
            except Exception as e:
                errors.append(f"Globals: {e}")

        if "agents" in data:
            try:
                for a in data["agents"]:
                    try:
                        blocks = a.get("prompt_blocks", [])
                        full_prompt = "\n".join(
                            pm.render(b.replace("$", "")) if b.startswith("$") else b
                            for b in blocks)
                        registry.register_agent(
                            name=a["name"], system_prompt=full_prompt,
                            tool_names=a.get("tools", []),
                            model_profile=a.get("model_profile"),
                            llm_overrides=a.get("llm_overrides"),
                            metadata=a.get("metadata"))
                    except Exception as e:
                        errors.append(f"Agent '{a.get('name', '?')}': {e}")
            except Exception as e:
                errors.append(f"Agents: {e}")

        if "workflow" in data:
            try:
                registry._workflow_cfg = data["workflow"]
            except Exception as e:
                errors.append(f"Workflow: {e}")

        print(f"[System] Configuration loaded from {filepath}")
        if errors:
            for err in errors:
                print(f"[Warning] {err}")
        return True

    @staticmethod
    def _make_tool(conf: dict) -> Optional[StructuredTool]:
        t = conf.get("type")
        if t == "python":
            scope = {}
            try:
                exec(conf.get("code", ""), {}, scope)
                fn = scope.get("run_tool")
                if fn:
                    return StructuredTool.from_function(
                        func=fn, name=conf["name"],
                        description=conf.get("description", ""))
            except Exception as e:
                print(f"[Error] Tool compile: {e}")
        elif t == "duckduckgo":
            try:
                from langchain_community.tools import DuckDuckGoSearchRun
                return DuckDuckGoSearchRun(name=conf["name"])
            except ImportError:
                print("[Error] duckduckgo-search not installed")
        return None