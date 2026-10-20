"""Anonymous Parallel tools using MCP's Streamable HTTP transport."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx
from langchain_core.tools import ToolException

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "xiaogpt (https://github.com/yihong0618/xiaogpt)"
REQUEST_TIMEOUT = 30
PROTOCOL_VERSION = "2025-03-26"
MAX_TOOL_CHARACTERS = 12000


@asynccontextmanager
async def _client(proxy):
    async with httpx.AsyncClient(
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/event-stream",
        },
        timeout=REQUEST_TIMEOUT,
        proxy=proxy,
    ) as client:
        try:
            yield client
        finally:
            if "Mcp-Session-Id" in client.headers:
                try:
                    async with asyncio.timeout(1):
                        await client.delete(MCP_URL)
                except (httpx.HTTPError, TimeoutError):
                    # Closing a server session is best effort, including servers
                    # that do not implement DELETE. Never retry the tool call.
                    pass


async def _request(
    client: httpx.AsyncClient, method: str, params: dict, request_id=None
):
    payload = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        payload["id"] = request_id
    # The endpoint may reply with JSON or SSE. Stop at the matching SSE reply
    # rather than waiting for a long-lived response stream to close.
    async with client.stream("POST", MCP_URL, json=payload) as response:
        response.raise_for_status()
        if method == "initialize":
            if session := response.headers.get("Mcp-Session-Id"):
                client.headers["Mcp-Session-Id"] = session
        if request_id is None:
            return None
        if (
            response.headers.get("content-type", "").split(";", 1)[0]
            == "text/event-stream"
        ):
            data = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line and data:
                    message = json.loads("\n".join(data))
                    data = []
                    if message.get("id") == request_id:
                        break
            else:
                raise ValueError("MCP stream ended without a response")
        else:
            await response.aread()
            message = response.json()
        if message.get("id") != request_id or "error" in message:
            raise ValueError("MCP request failed")
        return message["result"]


class ParallelSearch:
    def __init__(self, session_id: str, proxy: str | None = None):
        # This ID belongs to the bot's conversation, not an MCP connection.
        self.session_id = session_id
        self.proxy = proxy

    async def _call(self, name: str, arguments: dict) -> str:
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT):
                async with _client(self.proxy) as client:
                    initialized = await _request(
                        client,
                        "initialize",
                        {
                            "protocolVersion": PROTOCOL_VERSION,
                            "capabilities": {},
                            "clientInfo": {"name": "xiaogpt", "version": "1"},
                        },
                        1,
                    )
                    if initialized["protocolVersion"] != PROTOCOL_VERSION:
                        raise ValueError("Unsupported MCP protocol version")
                    client.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
                    await _request(client, "notifications/initialized", {})
                    available = await _request(client, "tools/list", {}, 2)
                    if name not in {tool["name"] for tool in available["tools"]}:
                        raise ToolException("Parallel tool is unavailable.")
                    result = await _request(
                        client,
                        "tools/call",
                        {
                            "name": name,
                            "arguments": {**arguments, "session_id": self.session_id},
                        },
                        3,
                    )
                    text = "\n".join(
                        item["text"]
                        for item in result["content"]
                        if item["type"] == "text"
                    )
                    if result.get("isError"):
                        raise ToolException("Parallel tool failed. Try again later.")
                    if len(text) > MAX_TOOL_CHARACTERS:
                        text = (
                            text[:MAX_TOOL_CHARACTERS]
                            + "\n[Truncated; narrow the query for more detail.]"
                        )
                    return text or "No results found."
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ToolException("Parallel request timed out. Try again later.") from exc
        except ToolException:
            raise
        except Exception as exc:
            # Never switch providers or retry a quota response behind the user's back.
            raise ToolException(
                "Parallel request failed. Try again later or select another provider."
            ) from exc

    async def search(self, query: str) -> str:
        return await self._call(
            "web_search", {"objective": query, "search_queries": [query]}
        )

    async def fetch(self, url: str) -> str:
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme in ("http", "https")
                and parsed.hostname
                and not parsed.username
                and not parsed.password
            )
        except ValueError:
            valid = False
        if not valid:
            raise ToolException(
                "Fetch requires an HTTP or HTTPS URL without credentials."
            )
        return await self._call("web_fetch", {"urls": [url]})
