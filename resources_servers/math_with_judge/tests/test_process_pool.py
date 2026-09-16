"""Regression tests for a symbolic comparison blocking the Gym event loop."""

import asyncio
import multiprocessing
import time

import pytest

from resources_servers.math_with_judge.process_pool import BoundedProcessPool
from resources_servers.math_with_judge.verification import verify_answer


GOLD = r"\(\sum_{t=1200}^{2000} \sum_{k=0}^{100}(-1)^k\binom{t-21k+99}{t-21k}\binom{100}{k}\)"
PRED = r"\boxed{21^{100} - \sum_{N=0}^{1199} \sum_{j=0}^{\lfloor N/21 \rfloor} (-1)^j \binom{100}{j} \binom{N - 21j + 99}{99}}"


@pytest.mark.asyncio
async def test_real_nested_sum_does_not_block_event_loop():
    pool = BoundedProcessPool(2, 20)
    ticks = 0
    running = True

    async def heartbeat():
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(0.02)

    task = asyncio.create_task(heartbeat())
    try:
        results = await asyncio.gather(
            pool.run(verify_answer, GOLD, PRED),
            pool.run(verify_answer, "4", r"\boxed{4}"),
        )
        assert results[0][0] == 0.0  # Existing math_verify comparison-timeout result.
        assert results[1] == (1.0, "4")
        assert ticks > 20
    finally:
        running = False
        await task
        await asyncio.to_thread(pool.close)


@pytest.mark.asyncio
async def test_hard_deadline_kills_workers_and_raises_instead_of_scoring():
    before = {p.pid for p in multiprocessing.active_children()}
    pool = BoundedProcessPool(1, 20)
    try:
        assert await pool.run(abs, -1) == 1
        pool._timeout_s = 0.1
        with pytest.raises(TimeoutError, match="process deadline"):
            await pool.run(time.sleep, 60)
        with pytest.raises(RuntimeError, match="process deadline"):
            await pool.run(abs, -1)
        assert {p.pid for p in multiprocessing.active_children()} <= before
    finally:
        await asyncio.to_thread(pool.close)


@pytest.mark.asyncio
async def test_cancelled_verification_releases_workers():
    before = {p.pid for p in multiprocessing.active_children()}
    pool = BoundedProcessPool(1, 20)
    try:
        assert await pool.run(abs, -1) == 1
        task = asyncio.create_task(pool.run(time.sleep, 60))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert {p.pid for p in multiprocessing.active_children()} <= before
    finally:
        await asyncio.to_thread(pool.close)


def test_http_server_remains_responsive_during_symbolic_comparison():
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import MagicMock

    from starlette.testclient import TestClient

    from nemo_gym.server_utils import ServerClient
    from resources_servers.math_with_judge.app import (
        LibraryJudgeMathResourcesServer,
        LibraryJudgeMathResourcesServerConfig,
    )

    config = LibraryJudgeMathResourcesServerConfig(
        name="math",
        host="127.0.0.1",
        port=1,
        entrypoint="",
        judge_model_server={"type": "responses_api_models", "name": "unused"},
        judge_responses_create_params={"input": []},
        should_use_judge=False,
        verifier_workers=2,
        verifier_timeout_s=20,
    )
    server = LibraryJudgeMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    body = {
        "question": "Count grade assignments.",
        "expected_answer": GOLD,
        "responses_create_params": {"input": []},
        "response": {
            "id": "test",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [
                {
                    "id": "answer",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": PRED, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
    }
    with TestClient(server.setup_webserver()) as client, ThreadPoolExecutor(1) as executor:
        verification = executor.submit(client.post, "/verify", json=body)
        time.sleep(0.1)
        started = time.monotonic()
        assert client.post("/seed_session", json={}).status_code == 200
        assert time.monotonic() - started < 2
        assert not verification.done()
        response = verification.result(timeout=20)
        assert response.status_code == 200, response.text
        assert response.json()["library_reward"] == 0.0
    assert server._process_pool._closed
