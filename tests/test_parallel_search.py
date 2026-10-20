"""Exercise the real agent and MCP transport with scripted inference and HTTP fixtures."""

import argparse
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, FunctionMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import ToolException
from pydantic import Field

from xiaogpt.bot import get_bot
from xiaogpt.bot.langchain_bot import LangChainBot
from xiaogpt.config import Config
from xiaogpt.langchain.chain import search_tools
from xiaogpt.langchain.parallel import MCP_URL, USER_AGENT, ParallelSearch


class RecordingModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = (
            self._generate(messages, stop, run_manager, **kwargs).generations[0].message
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content, additional_kwargs=message.additional_kwargs
            )
        )


def call(name, value):
    return AIMessage(
        content="",
        additional_kwargs={
            "function_call": {"name": name, "arguments": json.dumps({"__arg1": value})}
        },
    )


class MCPFixture:
    def __init__(self, *, sse=False, failure=None):
        self.requests = []
        self.sse = sse
        self.failure = failure
        self.started = asyncio.Event()

    async def respond(self, transport, request):
        if request.method == "DELETE":
            self.requests.append((request, {"method": "DELETE"}))
            return httpx.Response(204)
        payload = json.loads(request.content)
        self.requests.append((request, payload))
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ("web_search", "web_fetch")
                ]
            }
        elif method == "tools/call":
            self.started.set()
            if self.failure == "blocked":
                await asyncio.Event().wait()
            if self.failure == "http":
                return httpx.Response(503)
            if self.failure == "permission":
                return httpx.Response(403)
            if self.failure == "quota":
                return httpx.Response(429, headers={"Retry-After": "60"})
            if self.failure == "malformed":
                return httpx.Response(200, content="invalid JSON")
            text = "Official source https://example.org/news: new release on Monday."
            if self.failure == "large":
                text += "long page " * 5000
            result = {
                "content": [{"type": "text", "text": text}],
                "isError": self.failure == "tool",
            }
        else:
            raise AssertionError(method)
        body = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if self.sse:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"event: message\ndata: {json.dumps(body)}\n\n",
            )
        return httpx.Response(
            200, json=body, headers={"Mcp-Session-Id": "fixture-connection"}
        )

    def patch(self):
        fixture = self

        async def respond(transport, request):
            return await fixture.respond(transport, request)

        return patch.object(httpx.AsyncHTTPTransport, "handle_async_request", respond)

    @property
    def calls(self):
        return [p["params"] for _, p in self.requests if p["method"] == "tools/call"]


class ParallelAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    async def test_default_agent_search_fetch_followup_and_reset(self):
        fixture = MCPFixture(sse=True)
        model = RecordingModel(
            responses=[
                call("Search", "latest release news"),
                call("Fetch", "https://example.org/news"),
                AIMessage(content="The source says Monday."),
                call("Search", "release follow up"),
                AIMessage(content="Followup answer."),
                call("Search", "new conversation release"),
                AIMessage(content="Fresh answer."),
            ]
        )
        # A Parallel key in the environment must never authenticate this path.
        os.environ["PARALLEL_API_KEY"] = "must-not-be-used"
        bot = get_bot(Config(bot="langchain", openai_key="test"))
        with (
            fixture.patch(),
            patch("xiaogpt.langchain.chain.ChatOpenAI", return_value=model),
        ):
            self.assertEqual(
                await bot.ask("Find the release date"), "The source says Monday."
            )
            first_id = bot.search_session_id
            self.assertEqual(await bot.ask("Any followup?"), "Followup answer.")
            self.assertTrue(bot.has_history())
            bot.change_prompt("New conversation")
            self.assertNotEqual(bot.search_session_id, first_id)
            self.assertEqual(await bot.ask("Find it again"), "Fresh answer.")
        self.assertEqual(
            [c["name"] for c in fixture.calls],
            ["web_search", "web_fetch", "web_search", "web_search"],
        )
        self.assertEqual(
            [c["arguments"]["session_id"] for c in fixture.calls[:3]], [first_id] * 3
        )
        self.assertEqual(
            fixture.calls[3]["arguments"]["session_id"], bot.search_session_id
        )
        self.assertEqual(
            fixture.calls[0]["arguments"]["search_queries"], ["latest release news"]
        )
        self.assertEqual(
            fixture.calls[1]["arguments"]["urls"], ["https://example.org/news"]
        )
        observations = [
            m.content
            for messages in model.seen
            for m in messages
            if isinstance(m, FunctionMessage)
        ]
        self.assertTrue(
            any("https://example.org/news" in text for text in observations)
        )
        for request, _ in fixture.requests:
            self.assertEqual(str(request.url), MCP_URL)
            self.assertEqual(request.headers["user-agent"], USER_AGENT)
            self.assertNotIn("authorization", request.headers)

    async def test_independent_bots_interleave_and_retry_with_stable_ids(self):
        fixture = MCPFixture(failure="tool")

        def make_model(**kwargs):
            return RecordingModel(
                responses=[
                    call("Search", "news"),
                    AIMessage(content="Unable to search."),
                ]
            )

        a, b = LangChainBot("test", ""), LangChainBot("test", "")
        with (
            fixture.patch(),
            patch("xiaogpt.langchain.chain.ChatOpenAI", side_effect=make_model),
        ):
            await asyncio.gather(a.ask("first"), b.ask("second"))
            fixture.failure = None
            await a.ask("try again")
        ids = [c["arguments"]["session_id"] for c in fixture.calls]
        self.assertEqual(set(ids[:2]), {a.search_session_id, b.search_session_id})
        self.assertNotEqual(a.search_session_id, b.search_session_id)
        self.assertEqual(ids[-1], a.search_session_id)

    async def test_http_quota_tool_errors_and_no_fallback(self):
        for failure in ("http", "permission", "quota", "tool", "malformed"):
            with self.subTest(failure=failure):
                fixture = MCPFixture(failure=failure)
                model = RecordingModel(
                    responses=[
                        call("Search", "news"),
                        AIMessage(content="Search unavailable."),
                    ]
                )
                bot = LangChainBot("test", "saved-key", search_provider="parallel")
                with (
                    fixture.patch(),
                    patch("xiaogpt.langchain.chain.ChatOpenAI", return_value=model),
                    patch("xiaogpt.langchain.chain.SerpAPIWrapper") as serp,
                ):
                    self.assertEqual(await bot.ask("news"), "Search unavailable.")
                    serp.assert_not_called()
                self.assertEqual(len(fixture.calls), 1)
                self.assertTrue(
                    any(
                        isinstance(m, FunctionMessage) and "failed" in m.content
                        for messages in model.seen
                        for m in messages
                    )
                )

    async def test_session_cleanup_protocol_headers_and_bounded_output(self):
        fixture = MCPFixture(failure="large")
        model = RecordingModel(
            responses=[
                call("Fetch", "https://example.org/news"),
                AIMessage(content="Page summary."),
            ]
        )
        with (
            fixture.patch(),
            patch("xiaogpt.langchain.chain.ChatOpenAI", return_value=model),
        ):
            await LangChainBot("test", "").ask("Read the page")
        self.assertEqual(fixture.requests[-1][0].method, "DELETE")
        for request, payload in fixture.requests:
            if payload["method"] != "initialize":
                self.assertEqual(
                    request.headers["Mcp-Session-Id"], "fixture-connection"
                )
                self.assertEqual(request.headers["MCP-Protocol-Version"], "2025-03-26")
        observations = [
            m.content
            for messages in model.seen
            for m in messages
            if isinstance(m, FunctionMessage)
        ]
        self.assertTrue(
            any(
                "https://example.org/news" in text
                and "[Truncated;" in text
                and len(text) < 12100
                for text in observations
            )
        )

    async def test_successful_stream_consumes_search_observation(self):
        fixture = MCPFixture()
        model = RecordingModel(
            responses=[
                call("Search", "latest release news"),
                AIMessage(content="The release is Monday."),
            ]
        )
        with (
            fixture.patch(),
            patch("xiaogpt.langchain.chain.ChatOpenAI", return_value=model),
        ):
            bot = LangChainBot("test", "")
            answer = "".join(
                [part async for part in bot.ask_stream("Find the release")]
            )
        self.assertEqual(answer, "The release is Monday.")
        self.assertTrue(bot.has_history())
        self.assertEqual(len(fixture.calls), 1)
        self.assertTrue(
            any(
                isinstance(m, FunctionMessage)
                and "https://example.org/news" in m.content
                for messages in model.seen
                for m in messages
            )
        )

    async def test_timeout_and_cancellation_through_agent(self):
        for mode in ("timeout", "cancel", "stream_cancel"):
            with self.subTest(mode=mode):
                fixture = MCPFixture(failure="blocked")
                model = RecordingModel(
                    responses=[call("Search", "news"), AIMessage(content="Timed out.")]
                )
                bot = LangChainBot("test", "")
                with (
                    fixture.patch(),
                    patch("xiaogpt.langchain.chain.ChatOpenAI", return_value=model),
                    patch(
                        "xiaogpt.langchain.parallel.REQUEST_TIMEOUT",
                        0.1 if mode == "timeout" else 30,
                    ),
                ):
                    if mode == "timeout":
                        self.assertEqual(
                            await asyncio.wait_for(bot.ask("news"), 3), "Timed out."
                        )
                    else:

                        async def stream():
                            return [part async for part in bot.ask_stream("news")]

                        task = asyncio.create_task(
                            stream() if mode == "stream_cancel" else bot.ask("news")
                        )
                        await asyncio.wait_for(fixture.started.wait(), 3)
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(task, 3)
                self.assertEqual(len(fixture.calls), 1)

    async def test_closing_stream_cancels_pending_search(self):
        fixture = MCPFixture(failure="blocked")
        thought = call("Search", "news")
        thought.content = "Checking the web!"
        model = RecordingModel(responses=[thought, AIMessage(content="Done.")])
        with (
            fixture.patch(),
            patch("xiaogpt.langchain.chain.ChatOpenAI", return_value=model),
        ):
            stream = LangChainBot("test", "").ask_stream("Find news")
            self.assertEqual(
                await asyncio.wait_for(anext(stream), 3), "Checking the web!"
            )
            await asyncio.wait_for(fixture.started.wait(), 3)
            await asyncio.wait_for(stream.aclose(), 3)

    async def test_invalid_urls_never_reach_transport(self):
        fixture = MCPFixture()
        with fixture.patch():
            for url in (
                "file:///etc/passwd",
                "ftp://example.org",
                "https://user:pass@example.org",
                "https://",
                "https://[",
            ):
                with self.subTest(url=url), self.assertRaises(ToolException):
                    await ParallelSearch("test").fetch(url)
        self.assertEqual(fixture.requests, [])

    async def test_selection_credentials_overrides_and_disabled(self):
        wrapper = SimpleNamespace(run=lambda query: "SerpAPI result")
        with patch(
            "xiaogpt.langchain.chain.SerpAPIWrapper", return_value=wrapper
        ) as serp:
            self.assertEqual(
                search_tools("auto", "saved", "id")[0].func("q"), "SerpAPI result"
            )
            serp.assert_called_once_with(serpapi_api_key="saved")
            serp.reset_mock()
            self.assertEqual(
                [t.name for t in search_tools("parallel", "saved", "id")],
                ["Search", "Fetch"],
            )
            self.assertEqual(search_tools("none", "saved", "id"), [])
            serp.assert_not_called()
            with self.assertRaises(ValueError):
                search_tools("serpapi", "", "id")
            os.environ["SERPAPI_API_KEY"] = "environment-key"
            search_tools("auto", None, "id")
            serp.assert_called_once_with(serpapi_api_key="environment-key")
        model = RecordingModel(responses=[AIMessage(content="Offline answer.")])
        fixture = MCPFixture()
        with (
            fixture.patch(),
            patch("xiaogpt.langchain.chain.ChatOpenAI", return_value=model),
        ):
            bot = LangChainBot("test", "saved", search_provider="none")
            self.assertEqual(await bot.ask("hello"), "Offline answer.")
        self.assertEqual(fixture.requests, [])

    async def test_saved_serpapi_runs_through_agent_without_mcp(self):
        fixture = MCPFixture()
        search = SimpleNamespace(
            run=Mock(),
            arun=AsyncMock(
                return_value="Saved provider source https://example.org/news: Monday."
            ),
        )
        model = RecordingModel(
            responses=[
                call("Search", "release date"),
                AIMessage(content="Monday from SerpAPI."),
            ]
        )
        bot = get_bot(
            Config(bot="langchain", openai_key="test", serpapi_api_key="saved-key")
        )
        with (
            fixture.patch(),
            patch("xiaogpt.langchain.chain.ChatOpenAI", return_value=model),
            patch(
                "xiaogpt.langchain.chain.SerpAPIWrapper", return_value=search
            ) as wrapper,
        ):
            self.assertEqual(
                await bot.ask("Find the release date"), "Monday from SerpAPI."
            )
        wrapper.assert_called_once_with(serpapi_api_key="saved-key")
        search.arun.assert_awaited_once_with("release date")
        search.run.assert_not_called()
        self.assertEqual(fixture.requests, [])
        self.assertTrue(
            any(
                isinstance(m, FunctionMessage)
                and "Saved provider source https://example.org/news" in m.content
                for messages in model.seen
                for m in messages
            )
        )

    def test_saved_config_and_command_line_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(
                "bot: langchain\nopenai_key: test\nserpapi_api_key: saved\nlangchain_search_provider: none\n"
            )
            options = argparse.Namespace(
                config=str(path), langchain_search_provider=None
            )
            self.assertEqual(
                get_bot(Config.from_options(options)).search_provider, "none"
            )
            options.langchain_search_provider = "parallel"
            bot = get_bot(Config.from_options(options))
            self.assertEqual(bot.search_provider, "parallel")
            self.assertEqual(bot.serpapi_api_key, "saved")
        with self.assertRaises(ValueError):
            Config(bot="langchain", langchain_search_provider="invalid")
