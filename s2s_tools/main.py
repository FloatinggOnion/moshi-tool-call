#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .controller import Controller, LLMRouter
from .moshi_speech import MoshiSpeechInjector
from .tools import ToolRegistry
from .transcript import RollingTranscriptBuffer


def _format_turn(result) -> str:
    lines = []
    lines.append("states: " + " -> ".join(state.value for state in result.state_sequence))
    lines.append(f"route_source: {result.route_source}")
    lines.append("decision: " + json.dumps(asdict(result.decision), indent=2))
    if result.tool_result is not None:
        lines.append("tool_result: " + json.dumps(result.tool_result, indent=2))
    lines.append("spoken_text: " + result.spoken_text)
    return "\n".join(lines)


def _format_plan(plan) -> str:
    lines = []
    lines.append("states: " + " -> ".join(state.value for state in plan.state_sequence))
    lines.append(f"route_source: {plan.route_source}")
    lines.append("decision: " + json.dumps(asdict(plan.decision), indent=2))
    return "\n".join(lines)


async def run_once(text: str) -> None:
    registry = ToolRegistry()
    controller = Controller(LLMRouter(), registry)

    try:
        result = await controller.handle_input(text)
        print(_format_turn(result))
    finally:
        await controller.shutdown()


async def repl() -> None:
    registry = ToolRegistry()
    controller = Controller(LLMRouter(), registry)

    print("Type a user utterance. Enter 'quit' or 'exit' to stop.")

    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                print()
                break

            if not text:
                continue

            if text.lower() in {"quit", "exit"}:
                break

            result = await controller.handle_input(text)
            print(_format_turn(result))
            print()
    finally:
        await controller.shutdown()


async def transcript_dry_run() -> None:
    registry = ToolRegistry()
    controller = Controller(LLMRouter(), registry)
    buffer = RollingTranscriptBuffer()

    print("Paste rolling transcript snapshots one per line.")
    print("Use /final to force a flush, /reset to clear the buffer, or /exit to stop.")

    try:
        while True:
            try:
                snapshot = input("transcript> ")
            except EOFError:
                print()
                break

            command = snapshot.strip().lower()
            if command in {"/exit", "exit", "quit"}:
                break
            if command == "/reset":
                buffer.reset()
                print("buffer reset")
                continue
            if command == "/final":
                chunks = buffer.flush(final=True)
            else:
                chunks = buffer.observe(snapshot)

            if not chunks:
                print(f"pending: {buffer.pending!r}")
                continue

            for chunk in chunks:
                print(f"chunk({chunk.reason}): {chunk.text}")
                plan = await controller.plan_input(chunk.text)
                print(_format_plan(plan))
                print()
    finally:
        await controller.shutdown()


def render_test_speech(text: str, output_path: Path) -> None:
    injector = MoshiSpeechInjector()
    result = injector.render(text=text, output_path=output_path)
    print(f"rendered: {result.wav_path}")
    print(f"duration_seconds: {result.duration_seconds:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Typed-input controller demo for the s2s_tools project.")
    parser.add_argument("--text", help="Process a single utterance and exit.")
    parser.add_argument(
        "--transcript-dry-run",
        action="store_true",
        help="Inspect rolling transcript snapshots and show the LLM routing plan without executing tools.",
    )
    parser.add_argument(
        "--render-holding-phrase",
        action="store_true",
        help="Render 'Let me check that for you' as a Moshi TTS sample and write a wav file.",
    )
    parser.add_argument(
        "--render-final-answer",
        action="store_true",
        help="Render a canned final answer as a Moshi TTS sample and write a wav file.",
    )
    parser.add_argument(
        "--output-wav",
        default="outputs/moshi_speech.wav",
        help="Where to write the rendered wav when using a render flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render_holding_phrase:
        render_test_speech("Let me check that for you.", Path(args.output_wav))
        return
    if args.render_final_answer:
        render_test_speech("The current balance is 248,500 naira.", Path(args.output_wav))
        return
    if args.transcript_dry_run:
        asyncio.run(transcript_dry_run())
        return
    if args.text:
        asyncio.run(run_once(args.text))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
