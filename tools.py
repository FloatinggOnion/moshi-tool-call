from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True, frozen=True)
class ToolSpec:
	name: str
	description: str
	parameters: dict[str, Any]
	handler: ToolHandler


@dataclass(slots=True)
class ToolJob:
	tool_name: str
	args: dict[str, Any]
	future: asyncio.Future[dict[str, Any]]


def _json_request(url: str) -> dict[str, Any]:
	request = Request(url, headers={"User-Agent": "s2s-tools/0.1"})

	with urlopen(request, timeout=15) as response:
		return json.loads(response.read().decode("utf-8"))


def _get_current_date(_: dict[str, Any]) -> dict[str, Any]:
	now = datetime.now().astimezone()
	return {
		"date": now.date().isoformat(),
		"timezone": now.tzname(),
	}


def _get_current_time(_: dict[str, Any]) -> dict[str, Any]:
	now = datetime.now().astimezone()
	return {
		"time": now.strftime("%H:%M:%S"),
		"iso": now.isoformat(),
		"timezone": now.tzname(),
	}


def _get_account_balance(_: dict[str, Any]) -> dict[str, Any]:
	return {
		"account_id": "primary",
		"currency": "NGN",
		"available_balance": 248_500.75,
		"ledger_balance": 250_000.0,
		"as_of": datetime.now().astimezone().isoformat(),
	}


def _get_weather(args: dict[str, Any]) -> dict[str, Any]:
	location = str(args.get("location", "")).strip()

	if not location:
		raise ValueError("weather requires a location")

	geocode_url = (
		"https://geocoding-api.open-meteo.com/v1/search?"
		+ urlencode({"name": location, "count": 1, "language": "en", "format": "json"})
	)
	geocode_payload = _json_request(geocode_url)
	results = geocode_payload.get("results", [])

	if not results:
		raise ValueError(f"could not resolve location: {location}")

	place = results[0]
	forecast_url = (
		"https://api.open-meteo.com/v1/forecast?"
		+ urlencode(
			{
				"latitude": place["latitude"],
				"longitude": place["longitude"],
				"current": "temperature_2m,weather_code,wind_speed_10m",
				"timezone": "auto",
			}
		)
	)
	forecast_payload = _json_request(forecast_url)
	current = forecast_payload.get("current", {})

	return {
		"location": {
			"name": place.get("name"),
			"admin1": place.get("admin1"),
			"country": place.get("country"),
			"latitude": place.get("latitude"),
			"longitude": place.get("longitude"),
		},
		"current": {
			"temperature_c": current.get("temperature_2m"),
			"weather_code": current.get("weather_code"),
			"wind_speed_kmh": current.get("wind_speed_10m"),
			"time": current.get("time"),
		},
	}


class ToolRegistry:
	def __init__(self) -> None:
		self._tools: dict[str, ToolSpec] = {}
		self.register(
			ToolSpec(
				name="get_current_date",
				description="Return the current local date.",
				parameters={"type": "object", "properties": {}, "additionalProperties": False},
				handler=_get_current_date,
			)
		)
		self.register(
			ToolSpec(
				name="get_current_time",
				description="Return the current local time.",
				parameters={"type": "object", "properties": {}, "additionalProperties": False},
				handler=_get_current_time,
			)
		)
		self.register(
			ToolSpec(
				name="get_account_balance",
				description="Return a dummy account balance for controller testing.",
				parameters={"type": "object", "properties": {}, "additionalProperties": False},
				handler=_get_account_balance,
			)
		)
		self.register(
			ToolSpec(
				name="get_weather",
				description="Look up live weather data for a named location using Open-Meteo.",
				parameters={
					"type": "object",
					"properties": {
						"location": {"type": "string", "description": "City or place name"}
					},
					"required": ["location"],
					"additionalProperties": False,
				},
				handler=_get_weather,
			)
		)

	def register(self, spec: ToolSpec) -> None:
		self._tools[spec.name] = spec

	def get(self, name: str) -> ToolSpec:
		return self._tools[name]

	def names(self) -> list[str]:
		return list(self._tools)

	def specs(self) -> list[ToolSpec]:
		return list(self._tools.values())

	async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
		spec = self.get(name)
		return await asyncio.to_thread(spec.handler, args)


class AsyncToolQueue:
	def __init__(self, registry: ToolRegistry) -> None:
		self._registry = registry
		self._queue: asyncio.Queue[ToolJob] = asyncio.Queue()
		self._worker_task: asyncio.Task[None] | None = None

	async def start(self) -> None:
		if self._worker_task is None or self._worker_task.done():
			self._worker_task = asyncio.create_task(self._worker())

	async def submit(self, tool_name: str, args: dict[str, Any]) -> asyncio.Future[dict[str, Any]]:
		await self.start()
		future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
		await self._queue.put(ToolJob(tool_name=tool_name, args=args, future=future))
		return future

	async def shutdown(self) -> None:
		if self._worker_task is None:
			return

		self._worker_task.cancel()
		try:
			await self._worker_task
		except asyncio.CancelledError:
			pass

	async def _worker(self) -> None:
		while True:
			job = await self._queue.get()

			try:
				result = await self._registry.execute(job.tool_name, job.args)
			except Exception as exc:  # pragma: no cover - propagated to caller
				if not job.future.done():
					job.future.set_exception(exc)
			else:
				if not job.future.done():
					job.future.set_result(result)
			finally:
				self._queue.task_done()
