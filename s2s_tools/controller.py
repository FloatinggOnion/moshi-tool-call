from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.request import Request, urlopen

from .tools import AsyncToolQueue, ToolRegistry


class ControllerState(str, Enum):
    LISTENING = "LISTENING"
    INTENT_READY = "INTENT_READY"
    DIRECT = "DIRECT"
    HOLD_THEN_ANSWER = "HOLD_THEN_ANSWER"
    TOOL_PENDING = "TOOL_PENDING"
    RESULT_READY = "RESULT_READY"
    SPEAKING = "SPEAKING"


@dataclass(slots=True)
class RoutingDecision:
    intent: str
    tool_required: bool
    tool: str | None
    args: dict[str, Any]
    response_mode: str
    confidence: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoutingDecision":
        tool_required = bool(payload.get("tool_required", False))
        tool = payload.get("tool")
        # Small/weak routing models sometimes disagree between tool_required and
        # response_mode; tool_required is the more reliable signal, so response_mode
        # is derived from it rather than trusted independently.
        response_mode = "hold_then_answer" if tool_required and tool else "direct"

        return cls(
            intent=str(payload.get("intent", "unknown")),
            tool_required=tool_required and tool is not None,
            tool=tool,
            args=dict(payload.get("args", {}) or {}),
            response_mode=response_mode,
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(slots=True)
class TurnResult:
    input_text: str
    state_sequence: list[ControllerState]
    decision: RoutingDecision
    route_source: str
    tool_result: dict[str, Any] | None
    spoken_text: str


@dataclass(slots=True)
class TurnPlan:
    input_text: str
    state_sequence: list[ControllerState]
    decision: RoutingDecision
    route_source: str


class LLMRouter:
    """Routing controller backed by any OpenAI-compatible chat completions
    endpoint -- local (e.g. llama-server) by default, or Groq's cloud API
    when GROQ_API_URL/GROQ_API_KEY are set. Not tied to any specific model.
    """

    # Shared across all instances, not per-instance: when the router is a
    # local process (e.g. llama-server) co-located with Moshi on the same
    # GPU/CPU, overlapping requests compound resource contention (measured:
    # ~1.4x slower Moshi frame throughput under a single concurrent router
    # call). This queues router calls so at most one is in flight at a time,
    # regardless of how many Controller/LiveSession instances exist.
    _request_gate = asyncio.Semaphore(1)

    DEFAULT_LOCAL_URL = "http://localhost:8000/v1/chat/completions"

    def __init__(
        self,
        api_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        groq_url = os.getenv("GROQ_API_URL")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if api_url is not None:
            self.api_url = api_url
        elif groq_url:
            if not self.api_key:
                raise ValueError("GROQ_API_URL is set but GROQ_API_KEY is missing")
            self.api_url = groq_url
        else:
            # Local is the default modality; Groq only takes effect if
            # explicitly configured via the env vars above.
            self.api_url = self.DEFAULT_LOCAL_URL

        self.using_groq = self.api_url == groq_url and bool(groq_url)
        self.model_name = model_name or os.getenv(
            "LLM_MODEL", "llama-3.1-8b-instant" if self.using_groq else "local-model"
        )
        self.timeout_seconds = timeout_seconds

    def _build_messages(self, text: str, registry: ToolRegistry) -> list[dict[str, str]]:
        tool_lines = []
        for spec in registry.specs():
            tool_lines.append(f"- {spec.name}: {spec.description}")

        system_prompt = (
            "You are a deterministic routing controller for a voice agent.\n"
            "Return only one JSON object. No markdown, no code fences, no explanation.\n\n"
            "Schema:\n"
            '{"intent":"string","tool_required":true,"tool":"string or null","args":{},'
            '"response_mode":"direct or hold_then_answer","confidence":0.0}\n\n'
            "Rules:\n"
            "- Use tool_required false, tool null, args {}, response_mode direct when no tool is needed.\n"
            "- Use tool_required true and response_mode hold_then_answer when a tool is required.\n"
            "- For weather, use get_weather with a location string in args.\n"
            "- For date, time, and balance, use the named tool and empty args.\n\n"
            "Available tools:\n"
            + "\n".join(tool_lines)
            + "\n\nExample 1: user says \"What's the weather in Lagos\" -> "
            '{"intent":"weather","tool_required":true,"tool":"get_weather",'
            '"args":{"location":"Lagos"},"response_mode":"hold_then_answer","confidence":0.9}'
            + "\n\nExample 2: user says \"Tell me a joke\" -> "
            '{"intent":"small_talk","tool_required":false,"tool":null,'
            '"args":{},"response_mode":"direct","confidence":0.9}'
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

    def _call_model(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": messages,
            "max_tokens": 300,
        }
        headers = {"Content-Type": "application/json"}
        if self.using_groq:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            # llama-server-specific extension to suppress reasoning-model
            # <think> preambles; not part of the OpenAI schema Groq expects.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        request = Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )

        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))

        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _fallback_route(text: str) -> RoutingDecision:
        normalized = text.lower().strip()

        if any(
            phrase in normalized
            for phrase in (
                "tell me a joke",
                "who are you",
                "good morning",
                "speech-to-speech model",
                "friendly bank teller",
                "repeat after me",
                "what can you do",
                "last conversation",
                "plan my budget",
                "how much will i have in a year",
            )
        ):
            return RoutingDecision("small_talk", False, None, {}, "direct", 1.0)

        if "today's date" in normalized or "what's today's date" in normalized or normalized == "date":
            return RoutingDecision("get_date", True, "get_current_date", {}, "hold_then_answer", 1.0)

        if "what time" in normalized or "current time" in normalized or "right now" in normalized:
            return RoutingDecision("get_time", True, "get_current_time", {}, "hold_then_answer", 1.0)

        if "weather" in normalized or "raining" in normalized or "rain" in normalized:
            location = "Abuja" if "abuja" in normalized else "location"
            return RoutingDecision(
                "weather",
                True,
                "get_weather",
                {"location": location},
                "hold_then_answer",
                1.0,
            )

        if "account balance" in normalized or "how much do i have" in normalized or "broke" in normalized:
            return RoutingDecision("account_balance", True, "get_account_balance", {}, "hold_then_answer", 1.0)

        return RoutingDecision("small_talk", False, None, {}, "direct", 0.7)

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any]:
        # Reasoning models may leak draft JSON inside <think> blocks before their
        # final answer, so the LAST *top-level* parseable object wins. Only brace
        # positions at nesting depth 0 are tried, otherwise a nested value like
        # "args":{...} would be picked up as its own (incomplete) candidate.
        decoder = json.JSONDecoder()
        last_decoded: dict[str, Any] | None = None
        depth = 0

        for index, char in enumerate(raw_text):
            if char == "{":
                if depth == 0:
                    try:
                        decoded, _ = decoder.raw_decode(raw_text[index:])
                        if isinstance(decoded, dict):
                            last_decoded = decoded
                    except json.JSONDecodeError:
                        pass
                depth += 1
            elif char == "}":
                depth = max(0, depth - 1)

        if last_decoded is not None:
            return last_decoded

        return json.loads(raw_text)

    async def route(self, text: str, registry: ToolRegistry) -> tuple[RoutingDecision, str]:
        messages = self._build_messages(text, registry)
        try:
            async with self._request_gate:
                raw_text = await asyncio.to_thread(self._call_model, messages)
            payload = self._extract_json_object(raw_text)
            return RoutingDecision.from_dict(payload), "llm"
        except Exception:
            return self._fallback_route(text), "fallback"


class Controller:
    def __init__(self, router: LLMRouter, registry: ToolRegistry) -> None:
        self._router = router
        self._registry = registry
        self._tool_queue = AsyncToolQueue(registry)
        self.state = ControllerState.LISTENING

    def _transition(self, new_state: ControllerState, log: list[ControllerState]) -> None:
        self.state = new_state
        log.append(new_state)

    async def plan_input(self, text: str) -> TurnPlan:
        state_sequence = [ControllerState.LISTENING]
        self.state = ControllerState.LISTENING

        self._transition(ControllerState.INTENT_READY, state_sequence)
        decision, route_source = await self._router.route(text, self._registry)

        if not decision.tool_required or decision.response_mode == "direct":
            self._transition(ControllerState.DIRECT, state_sequence)
            self._transition(ControllerState.SPEAKING, state_sequence)
        else:
            self._transition(ControllerState.HOLD_THEN_ANSWER, state_sequence)
            self._transition(ControllerState.TOOL_PENDING, state_sequence)
            self._transition(ControllerState.RESULT_READY, state_sequence)
            self._transition(ControllerState.SPEAKING, state_sequence)

        return TurnPlan(
            input_text=text,
            state_sequence=state_sequence,
            decision=decision,
            route_source=route_source,
        )

    async def handle_input(self, text: str) -> TurnResult:
        plan = await self.plan_input(text)
        state_sequence = plan.state_sequence
        decision = plan.decision
        route_source = plan.route_source

        if not decision.tool_required or decision.response_mode == "direct":
            self._transition(ControllerState.DIRECT, state_sequence)
            spoken_text = f"Direct response for intent '{decision.intent}'."
            self._transition(ControllerState.SPEAKING, state_sequence)
            return TurnResult(
                input_text=text,
                state_sequence=state_sequence,
                decision=decision,
                route_source=route_source,
                tool_result=None,
                spoken_text=spoken_text,
            )

        self._transition(ControllerState.HOLD_THEN_ANSWER, state_sequence)
        self._transition(ControllerState.TOOL_PENDING, state_sequence)

        if not decision.tool:
            raise ValueError("tool_required was true but no tool name was returned")

        future = await self._tool_queue.submit(decision.tool or "", decision.args)
        tool_result = await future
        self._transition(ControllerState.RESULT_READY, state_sequence)

        spoken_text = f"Tool '{decision.tool}' returned result."
        self._transition(ControllerState.SPEAKING, state_sequence)
        return TurnResult(
            input_text=text,
            state_sequence=state_sequence,
            decision=decision,
            route_source=route_source,
            tool_result=tool_result,
            spoken_text=spoken_text,
        )

    async def shutdown(self) -> None:
        await self._tool_queue.shutdown()
