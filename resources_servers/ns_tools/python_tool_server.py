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

"""Launch the upstream Python MCP app with Gym request diagnostics."""

import argparse
import logging

import uvicorn
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.servers import python_tool

from nemo_gym.diagnostics import install_server_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sandbox-host", default="127.0.0.1")
    parser.add_argument("--sandbox-port", default="6000")
    parser.add_argument("--disable-session-restore", action="store_true")
    args = parser.parse_args()
    python_tool.sandbox = get_sandbox(
        sandbox_type="local",
        host=args.sandbox_host,
        port=args.sandbox_port,
        disable_session_restore=args.disable_session_restore,
    )
    for name in ("mcp", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)
    app = python_tool.mcp.streamable_http_app()
    install_server_diagnostics(app, "python_tool")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
