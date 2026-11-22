# model_manager.py
# Model profiles with resource-aware LLM creation

from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from resource_manager import ResourceManager


class SmartModelManager:
    def __init__(self, resource_manager: ResourceManager = None):
        self.profiles: Dict[str, Dict] = {}
        self.current_profile_name: str = "default"
        self.rm = resource_manager or ResourceManager()
        self.rm.start()
        self.load_profiles({
            "default": {"model_name": "qwen2.5:0.5b", "temperature": 0.0}
        }, "default")

    def load_profiles(self, profiles: Dict, default_name: str):
        self.profiles = profiles
        self.current_profile_name = default_name if default_name in profiles else "default"
        print(f"[ModelManager] Loaded {len(profiles)} profiles. Active: {self.current_profile_name}")

    def set_mode(self, mode: str) -> bool:
        if mode not in self.profiles:
            print(f"[ModelManager] Error: Profile '{mode}' not found.")
            return False
        model = self.profiles[mode].get("model_name", "")
        check = self.rm.can_load(model)
        if not check["can_load"]:
            print(f"[ModelManager] Cannot load {model}: {'; '.join(check['warnings'])}")
            return False
        self.current_profile_name = mode
        print(f"[ModelManager] Switched to {mode} ({check['mode']})")
        return True

    def get_llm(self, override_profile: Optional[str] = None,
                llm_overrides: Optional[Dict] = None) -> ChatOpenAI:
        pname = override_profile or self.current_profile_name
        if pname not in self.profiles:
            pname = "default"
        cfg = self.profiles[pname]
        opts = {
            "temperature": cfg.get("temperature", 0.0),
            "top_p": cfg.get("top_p", 1.0),
            "max_tokens": min(cfg.get("max_tokens", 2048), 4096),
        }
        if llm_overrides:
            for k, v in llm_overrides.items():
                if k in {"temperature", "top_p", "frequency_penalty",
                         "presence_penalty", "max_tokens"}:
                    opts[k] = v
        return ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="dummy-key",
            model=cfg.get("model_name", "qwen2.5:0.5b"),
            **opts)

    def get_resource_status(self) -> Dict:
        return self.rm.status()

    def auto_select(self) -> str:
        suggested = self.rm.suggest_model()
        for pname, cfg in self.profiles.items():
            if cfg.get("model_name") == suggested:
                self.set_mode(pname)
                return pname
        return self.current_profile_name

    @property
    def current_model_name(self) -> str:
        return self.profiles.get(self.current_profile_name, {}).get(
            "model_name", "qwen2.5:0.5b")