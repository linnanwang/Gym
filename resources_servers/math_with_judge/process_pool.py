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

"""Bounded process execution for CPU verifiers that must not block an HTTP loop."""

import asyncio
import multiprocessing
import os
from typing import Any, Callable, Optional


class BoundedProcessPool:
    def __init__(self, workers: int, timeout_s: float) -> None:
        # Each child runs verification on its main thread, where signal-based
        # library deadlines work. Bound numerical-library parallelism per child.
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        self._pool = multiprocessing.get_context("spawn").Pool(workers)
        self._slots = asyncio.Semaphore(workers)
        self._timeout_s = timeout_s
        self._failure: Optional[str] = None
        self._closed = False

    async def run(self, function: Callable[..., Any], *args: Any) -> Any:
        async with self._slots:
            if self._failure is not None:
                raise RuntimeError(self._failure)
            if self._closed:
                raise RuntimeError("Verification pool is closed")
            result = self._pool.apply_async(function, args)
            try:
                return await asyncio.to_thread(result.get, self._timeout_s)
            except multiprocessing.TimeoutError as error:
                self._failure = f"Verification exceeded the {self._timeout_s}s process deadline"
                await asyncio.to_thread(self.close)
                raise TimeoutError(self._failure) from error
            except asyncio.CancelledError:
                self._failure = "Verification request was cancelled"
                await asyncio.to_thread(self.close)
                raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.terminate()
            self._pool.join()
