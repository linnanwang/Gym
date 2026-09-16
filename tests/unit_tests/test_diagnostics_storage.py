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

"""Diagnostic filesystem failures must not prevent serving requests."""

import asyncio
import errno
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from nemo_gym import diagnostics


@pytest.mark.parametrize("failure", ["mkdir", "open"])
def test_startup_continues_when_diagnostics_storage_fails(monkeypatch, tmp_path, caplog, failure):
    app = FastAPI()
    monkeypatch.setenv("NRL_GYM_DIAGNOSTICS_DIR", str(tmp_path))
    error = OSError(errno.EDQUOT, "Disk quota exceeded")
    with monkeypatch.context() as patch:
        if failure == "mkdir":
            patch.setattr(Path, "mkdir", Mock(side_effect=error))
        else:
            patch.setattr(diagnostics, "open", Mock(side_effect=error), raising=False)
        diagnostics.install_server_diagnostics(app, "test")
    assert not app.user_middleware
    assert "Disabling diagnostics" in caplog.text
    with TestClient(app) as client:
        assert client.get("/").status_code == 404


@pytest.mark.asyncio
async def test_snapshot_write_failure_does_not_break_lifespan(monkeypatch, tmp_path, caplog):
    app = FastAPI()
    monkeypatch.setenv("NRL_GYM_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback_later", Mock())
    cancel = Mock()
    monkeypatch.setattr(diagnostics.faulthandler, "cancel_dump_traceback_later", cancel)
    diagnostics.install_server_diagnostics(app, "test")
    monkeypatch.setattr(Path, "write_text", Mock(side_effect=OSError(errno.EDQUOT, "Disk quota exceeded")))
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.01)
        assert "Stopping diagnostic snapshots" in caplog.text
    cancel.assert_called_once()
