"""Prompt reader for LivingMemory's last completed recall snapshot."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from astrbot.core.prompt.extensions.types import PromptExtension
from astrbot.core.prompt.interfaces import PromptExtensionCollectorInterface

from .utils import format_memories_for_injection

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import ProviderRequest
    from astrbot.api.star import Context


_PROMPT_CONTEXT_VIEWS = frozenset(
    {"turn_plan", "olv_stream", "final_result", "proactive"}
)


class LivingMemoryPromptCacheCollector(PromptExtensionCollectorInterface):
    """Expose only the completed LivingMemory snapshot to Persona.

    ``collect_prompt_messages`` is intentionally synchronous: it reads the
    in-process prompt cache and returns immediately on a miss. Retrieval and
    cache refresh are scheduled by the plugin's background writer path.
    """

    prompt_message_context_views = _PROMPT_CONTEXT_VIEWS

    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin

    @property
    def plugin_id(self) -> str:
        return "livingmemory_prompt_recall_snapshot"

    @property
    def priority(self) -> int:
        return 20

    @property
    def prompt_message_targets(self) -> frozenset[str]:
        return frozenset({"persona"})

    @property
    def prompt_message_placement(self) -> str:
        return "after_user"

    @property
    def lifecycle(self) -> str:
        return "dynamic"

    async def collect(
        self,
        event: AstrMessageEvent,
        plugin_context: Context,
        config: Any,
        provider_request: ProviderRequest | None = None,
    ) -> list[PromptExtension]:
        del event, plugin_context, config, provider_request
        return []

    def collect_prompt_messages(
        self,
        event: AstrMessageEvent,
        plugin_context: Context,
        *,
        target: str,
        context_view: str | None = None,
        provider_request: ProviderRequest | None = None,
    ) -> list[dict[str, Any]]:
        del plugin_context
        if target != "persona":
            return []
        if (
            context_view is not None
            and context_view not in self.prompt_message_context_views
        ):
            return []

        initializer = getattr(self._plugin, "initializer", None)
        engine = getattr(initializer, "memory_engine", None)
        getter = getattr(engine, "get_prompt_recall_snapshot", None)
        if not callable(getter):
            return []

        session_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if not session_id:
            return []
        persona_id = _resolve_persona_id(event, provider_request)
        try:
            results = getter(session_id, persona_id, limit=5)
        except Exception:
            return []
        if not results:
            return []

        memories = [
            {
                "id": getattr(result, "doc_id", None),
                "content": result.content,
                "score": result.final_score,
                "metadata": result.metadata,
                "timestamp": result.metadata.get("create_time"),
            }
            for result in results
        ]
        text = format_memories_for_injection(memories)
        return [{"role": "user", "content": text}] if text else []


def _resolve_persona_id(
    event: AstrMessageEvent,
    provider_request: ProviderRequest | None,
) -> str | None:
    conversation = getattr(provider_request, "conversation", None)
    conversation_persona_id = getattr(conversation, "persona_id", None)
    if isinstance(conversation_persona_id, str) and conversation_persona_id.strip():
        return conversation_persona_id.strip()

    try:
        from astrbot.core.interaction.turn_state import get_interaction_turn_state

        state = get_interaction_turn_state(event)
        state_persona_id = getattr(state, "persona_id", None)
        if isinstance(state_persona_id, str) and state_persona_id.strip():
            return state_persona_id.strip()
    except Exception:
        pass

    config = event.get_extra("_astrbot_config", {})
    if isinstance(config, dict):
        provider_settings = config.get("provider_settings", {})
        if isinstance(provider_settings, dict):
            default_personality = provider_settings.get("default_personality")
            if isinstance(default_personality, str) and default_personality.strip():
                return default_personality.strip()
    return None


__all__ = ["LivingMemoryPromptCacheCollector"]
