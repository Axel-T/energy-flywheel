"""
energy_context_pipeline.py

OpenWebUI pipeline that fetches the current sensor context snapshot from
the Node-RED context API and prepends it to every user message before it
reaches the language model.

Install in OpenWebUI:
  Admin Panel → Pipelines → Upload a pipeline → select this file → Enable

Configuration (edit the Valves class or set via the OpenWebUI UI):
  CONTEXT_API_URL — URL of the Node-RED /api/context endpoint
  CONTEXT_TIMEOUT — seconds to wait before giving up (default: 5)

If the context API is unreachable, the pipeline passes the message through
unchanged rather than blocking the user's query.
"""

import json
from typing import List, Optional

import httpx


class Pipeline:

    class Valves:
        CONTEXT_API_URL: str = "http://metrics-server-ip:1880/api/context"
        CONTEXT_TIMEOUT: int = 5

    def __init__(self):
        self.name   = "Energy context injector"
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"[energy-context] Pipeline started")
        print(f"[energy-context] Context API: {self.valves.CONTEXT_API_URL}")

    async def on_shutdown(self):
        print("[energy-context] Pipeline stopped")

    async def inlet(
        self,
        body: dict,
        user: Optional[dict] = None,
    ) -> dict:
        """Fetch sensor context and prepend it to the latest user message."""

        # Fetch current sensor snapshot from Node-RED
        context_block = ""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.valves.CONTEXT_API_URL,
                    timeout=self.valves.CONTEXT_TIMEOUT,
                )
                response.raise_for_status()
                context_data = response.json()
                context_block = (
                    "Current sensor readings from the home energy system:\n"
                    + json.dumps(context_data, indent=2, ensure_ascii=False)
                    + "\n\n"
                )
        except httpx.TimeoutException:
            print("[energy-context] Warning: context API timed out — "
                  "passing message without context")
        except Exception as e:
            print(f"[energy-context] Warning: context API error ({e}) — "
                  "passing message without context")

        if not context_block:
            return body

        # Prepend context to the most recent user message
        messages: List[dict] = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                original         = msg.get("content", "")
                msg["content"]   = context_block + original
                break

        return body
