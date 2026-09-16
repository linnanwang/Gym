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

"""Opt-in diagnostics for long-running Gym HTTP requests and blocked event loops."""

import asyncio
import contextlib
import faulthandler
import json
import logging
import os
import socket
import time
import traceback
from collections import Counter
from pathlib import Path


def install_server_diagnostics(app, name):
    """Persist pending requests/tasks without depending on a training-step boundary."""
    directory = os.environ.get("NRL_GYM_DIAGNOSTICS_DIR")
    if not directory:
        return
    root = Path(directory)
    prefix = root / f"{socket.gethostname()}-{name}-{os.getpid()}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        stacks = open(str(prefix) + ".stacks", "a", buffering=1)
    except OSError as error:
        logging.getLogger(__name__).warning("Disabling diagnostics for %s: %s", name, error)
        return
    # The C watchdog also works when synchronous verifier code holds the GIL.
    faulthandler.dump_traceback_later(180, repeat=True, file=stacks)
    pending = {}
    completed = Counter()

    class RequestDiagnostics:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.app(scope, receive, send)
            task = asyncio.current_task()
            key = id(task)
            path = f"{scope['method']} {scope['path']}"
            pending[key] = (time.monotonic(), path, scope)
            try:
                await self.app(scope, receive, send)
            finally:
                pending.pop(key, None)
                completed[path] += 1

    app.add_middleware(RequestDiagnostics)

    async def snapshot_loop():
        while True:
            now = time.monotonic()
            requests = [
                {"task_id": key, "age_s": round(now - started, 1), "path": path, "session": scope.get("session", {})}
                for key, (started, path, scope) in list(pending.items())
            ]
            tasks = []
            for task in asyncio.all_tasks():
                frames = []
                awaited = task.get_coro()
                seen = set()
                while awaited is not None and id(awaited) not in seen:
                    seen.add(id(awaited))
                    frame = getattr(awaited, "cr_frame", None) or getattr(awaited, "gi_frame", None)
                    if frame is not None:
                        frames.extend(traceback.format_stack(frame, limit=1))
                    awaited = getattr(awaited, "cr_await", None) or getattr(awaited, "gi_yieldfrom", None)
                tasks.append({"id": id(task), "task": repr(task), "frames": frames})
            data = {
                "time": time.time(),
                "pid": os.getpid(),
                "server": name,
                "pending": requests,
                "completed": dict(completed),
                "tasks": tasks,
            }
            temporary = Path(str(prefix) + ".json.tmp")
            try:
                temporary.write_text(json.dumps(data, default=str))
                temporary.replace(str(prefix) + ".json")
            except OSError as error:
                logging.getLogger(__name__).warning("Stopping diagnostic snapshots for %s: %s", name, error)
                return
            await asyncio.sleep(30)

    original_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application) as state:
            task = asyncio.create_task(snapshot_loop())
            try:
                yield state
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                faulthandler.cancel_dump_traceback_later()
                stacks.close()

    app.router.lifespan_context = lifespan
