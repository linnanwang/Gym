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

"""Pure math verification shared by synchronous and isolated workers."""

import contextlib
import logging
from functools import lru_cache
from io import StringIO
from typing import Optional

from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


@contextlib.contextmanager
def mute_output():
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        yield


@lru_cache(maxsize=1)
def get_library_verifier():
    logging.getLogger("math_verify").setLevel(logging.CRITICAL)
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def strip_math_delimiters(s: str) -> str:
    """Strip outer math delimiters from expected answers.

    Many expected_answer values are wrapped in \\(...\\) or $...$,
    which causes the math_verify parser to fail when we wrap them
    in \\boxed{}.  Removing these outer delimiters fixes parsing.
    """
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    return s


def verify_with_library(verifier, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
    # This functionality is migrated from Nemo RL.
    # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
    try:
        stripped = strip_math_delimiters(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"
        with mute_output():
            ret_score, extracted_answer = verifier([ground_truth_parsable], [generated_answer])

        reward = float(ret_score)

        if extracted_answer is not None:
            # Make sure the extracted answer has two elements.
            assert len(extracted_answer) == 2

            extracted_gold, extracted_prediction = extracted_answer

            # Get the extracted answer.
            for pred in extracted_prediction:
                if any(grader.verify(gold, pred) for gold in extracted_gold):
                    extracted_answer = pred
                    break
            else:
                # If no match is found, that means all the answers are
                # incorrect.  The first prediction is used as the extracted
                # answer.
                extracted_answer = extracted_prediction[0] if extracted_prediction else None

        return reward, extracted_answer

    # It's possible to emit a TimeoutException and that wouldn't be caught since
    # it actually subclasses from BaseException and math-verify itself does not
    # catch it.
    except (Exception, TimeoutException):
        return 0.0, None


def verify_answer(expected_answer: str, generated_answer: str):
    return verify_with_library(get_library_verifier(), expected_answer, generated_answer)
