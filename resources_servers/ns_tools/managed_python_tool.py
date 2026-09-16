# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PythonTool session cleanup using the sandbox's session-affine DELETE API."""

import asyncio
import logging

import httpx
from nemo_skills.mcp.servers.python_tool import PythonTool as BasePythonTool
from nemo_skills.mcp.tool_manager import ToolManager as BaseToolManager


class PythonTool(BasePythonTool):
    def __init__(self, base_url=None):
        super().__init__(base_url)
        self._sandbox_url = None
        self._cleanup_client = None

    def configure(self, overrides=None, context=None):
        super().configure(overrides, context)
        sandbox = context["sandbox"]
        self._sandbox_url = f"http://{sandbox['host']}:{sandbox['port']}"
        self._cleanup_client = httpx.AsyncClient(timeout=10)

    async def execute(self, tool_name, arguments, extra_args=None):
        request_id = extra_args["request_id"]
        previous_session = self.requests_to_sessions.get(request_id)
        try:
            # Let the sandbox allocate first-call IDs: supplying one here makes
            # upstream report a false state-restoration failure to the model.
            return await super().execute(tool_name, arguments, dict(extra_args))
        finally:
            # A transport error may return no ID; retain a known ID for cleanup.
            if previous_session and not self.requests_to_sessions.get(request_id):
                self.requests_to_sessions[request_id] = previous_session

    async def cleanup_session(self, request_id):
        session_id = self.requests_to_sessions.get(request_id)
        if session_id is None:
            return
        async with asyncio.timeout(15):
            for attempt in range(3):
                try:
                    response = await self._cleanup_client.delete(
                        f"{self._sandbox_url}/sessions/{session_id}",
                        headers={"X-Session-ID": str(session_id)},
                    )
                    if response.status_code != 404:
                        response.raise_for_status()
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if attempt == 2 or not (status in (408, 429) or 500 <= status < 600):
                        raise
                else:
                    # Keep the ID until deletion is confirmed, including when
                    # a previous DELETE succeeded but its response was lost.
                    self.requests_to_sessions.pop(request_id, None)
                    logging.info("Deleted sandbox session %s", session_id)
                    return
                await asyncio.sleep(0.5)

    async def shutdown(self):
        await super().shutdown()
        if self._cleanup_client is not None:
            await self._cleanup_client.aclose()


class ToolManager(BaseToolManager):
    async def cleanup_session(self, request_id):
        for tool in self._tools.values():
            if isinstance(tool, PythonTool):
                await tool.cleanup_session(request_id)
