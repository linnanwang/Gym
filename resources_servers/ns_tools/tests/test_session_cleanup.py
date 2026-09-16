"""Session cleanup regression tests without a live model or sandbox."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from resources_servers.ns_tools.managed_python_tool import PythonTool
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


@pytest.mark.asyncio
async def test_cleanup_uses_same_session_and_releases_mapping():
    tool = PythonTool()
    tool._sandbox_url = "http://sandbox:6000"
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    tool._cleanup_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    tool.requests_to_sessions["rollout"] = "ipython-session"
    await tool.cleanup_session("rollout")
    await tool.cleanup_session("rollout")
    assert len(requests) == 1
    assert requests[0].url.path == "/sessions/ipython-session"
    assert requests[0].headers["X-Session-ID"] == "ipython-session"
    assert not tool.requests_to_sessions
    await tool.shutdown()


@pytest.mark.asyncio
async def test_first_call_preserves_upstream_session_creation():
    tool = PythonTool()
    tool._client = MagicMock(
        call_tool=AsyncMock(
            return_value={
                "session_id": "allocated",
                "output_dict": {"stdout": "42\n", "stderr": ""},
            }
        )
    )
    assert await tool.execute("stateful_python_code_exec", {"code": "print(42)"}, {"request_id": "r"}) == "42"
    assert tool._client.call_tool.call_args.kwargs["extra_args"]["session_id"] is None
    assert tool.requests_to_sessions["r"] == "allocated"


@pytest.mark.asyncio
async def test_transport_error_retains_known_session_for_cleanup():
    tool = PythonTool()
    tool.requests_to_sessions["r"] = "known"
    tool._client = MagicMock(
        call_tool=AsyncMock(
            return_value={
                "session_id": None,
                "output_dict": {"stdout": "", "stderr": "transport failed"},
            }
        )
    )
    await tool.execute("stateful_python_code_exec", {"code": "print(1)"}, {"request_id": "r"})
    assert tool.requests_to_sessions["r"] == "known"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("generation failed"), asyncio.CancelledError()])
async def test_agent_cleans_session_after_failure_or_cancellation(failure):
    seed = MagicMock(cookies={"resource-session": "original"})
    cleanup = MagicMock(read=AsyncMock())
    client = MagicMock(post=AsyncMock(side_effect=[seed, failure, cleanup]))
    config = SimpleAgentConfig(
        name="agent",
        host="127.0.0.1",
        port=1,
        entrypoint="",
        resources_server={"type": "resources_servers", "name": "ns_tools"},
        model_server={"type": "responses_api_models", "name": "model"},
        cleanup_session=True,
    )
    agent = SimpleAgent.model_construct(config=config, server_client=client)
    with patch("responses_api_agents.simple_agent.app.raise_for_status", new=AsyncMock()):
        with pytest.raises(type(failure)):
            await agent.run(MagicMock(cookies={}), MagicMock())
    assert client.post.call_args.kwargs["url_path"] == "/cleanup_session"
    assert client.post.call_args.kwargs["cookies"] == seed.cookies


@pytest.mark.asyncio
@pytest.mark.parametrize("verify_fails", [False, True])
async def test_agent_cleanup_after_verification(verify_fails):
    seed = MagicMock(cookies={"resource-session": "seed"})
    generated = MagicMock(cookies={"resource-session": "generated"})
    verified = MagicMock()
    cleanup = MagicMock(read=AsyncMock())
    failure = RuntimeError("verification failed")
    client = MagicMock(post=AsyncMock(side_effect=[seed, generated, failure if verify_fails else verified, cleanup]))
    config = SimpleAgentConfig(
        name="agent",
        host="127.0.0.1",
        port=1,
        entrypoint="",
        resources_server={"type": "resources_servers", "name": "ns_tools"},
        model_server={"type": "responses_api_models", "name": "model"},
        cleanup_session=True,
    )
    agent = SimpleAgent.model_construct(config=config, server_client=client)
    body = MagicMock()
    body.model_dump.return_value = {}
    result = object()
    with (
        patch("responses_api_agents.simple_agent.app.raise_for_status", new=AsyncMock()),
        patch("responses_api_agents.simple_agent.app.get_response_json", new=AsyncMock(return_value={})),
        patch("responses_api_agents.simple_agent.app.SimpleAgentVerifyRequest.model_validate"),
        patch("responses_api_agents.simple_agent.app.SimpleAgentVerifyResponse.model_validate", return_value=result),
    ):
        if verify_fails:
            with pytest.raises(RuntimeError, match="verification failed"):
                await agent.run(MagicMock(cookies={}), body)
        else:
            assert await agent.run(MagicMock(cookies={}), body) is result
    assert client.post.call_args.kwargs["url_path"] == "/cleanup_session"
    assert client.post.call_args.kwargs["cookies"] == generated.cookies


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [httpx.ConnectError("unavailable"), 408, 429, 503])
@pytest.mark.parametrize("success_status", [200, 404])
async def test_cleanup_retries_transient_failure_before_releasing_mapping(failure, success_status):
    tool = PythonTool()
    tool._sandbox_url = "http://sandbox:6000"
    tool.requests_to_sessions["rollout"] = "session"
    requests = []

    def handle(request):
        requests.append(request)
        assert tool.requests_to_sessions["rollout"] == "session"
        assert request.url.path == "/sessions/session"
        assert request.headers["X-Session-ID"] == "session"
        if len(requests) == 1:
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure)
        return httpx.Response(success_status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        tool._cleanup_client = client
        await tool.cleanup_session("rollout")
        assert len(requests) == 2
        assert "rollout" not in tool.requests_to_sessions


@pytest.mark.asyncio
async def test_cleanup_exhausted_retries_retain_session_for_later_cleanup():
    tool = PythonTool()
    tool._sandbox_url = "http://sandbox:6000"
    tool.requests_to_sessions["rollout"] = "session"
    requests = []
    available = False

    def handle(request):
        requests.append(request)
        if not available:
            raise httpx.ConnectError("unavailable", request=request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        tool._cleanup_client = client
        with pytest.raises(httpx.ConnectError):
            await tool.cleanup_session("rollout")
        assert len(requests) == 3
        assert tool.requests_to_sessions["rollout"] == "session"
        available = True
        await tool.cleanup_session("rollout")
        assert len(requests) == 4
        assert "rollout" not in tool.requests_to_sessions


@pytest.mark.asyncio
async def test_cleanup_permanent_failure_retains_session_without_retry():
    tool = PythonTool()
    tool._sandbox_url = "http://sandbox:6000"
    tool.requests_to_sessions["rollout"] = "session"
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(403)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        tool._cleanup_client = client
        with pytest.raises(httpx.HTTPStatusError):
            await tool.cleanup_session("rollout")
        assert len(requests) == 1
        assert tool.requests_to_sessions["rollout"] == "session"


@pytest.mark.asyncio
async def test_cleanup_cancellation_retains_session():
    tool = PythonTool()
    tool._sandbox_url = "http://sandbox:6000"
    tool.requests_to_sessions["rollout"] = "session"
    started = asyncio.Event()

    async def handle(request):
        started.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        tool._cleanup_client = client
        task = asyncio.create_task(tool.cleanup_session("rollout"))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert tool.requests_to_sessions["rollout"] == "session"
