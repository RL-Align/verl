# SPDX-License-Identifier: Apache-2.0
"""Pinned native Ray AF_UNIX path validation, not real rollout acceptance."""

import ast
import inspect
from pathlib import Path
from uuid import uuid4

from ray._private.utils import validate_socket_filepath

from verl.experimental.rl_kernel import phase_runtime


def test_actual_phase_ray_directory_fits_native_socket_limit_and_is_private():
    function = ast.parse(inspect.getsource(phase_runtime.rollout_phase))
    init = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "ray"
        and n.func.attr == "init"
    )
    value = next(k.value for k in init.keywords if k.arg == "_temp_dir")
    expression = compile(ast.Expression(value), "actual-phase-ray-directory", "eval")
    paths = [eval(expression, {"uuid4": uuid4}) for _ in range(8)]
    assert len(set(paths)) == len(paths)
    for root in paths:
        assert Path(root).parent == Path("/tmp/rv4")
        socket = root + "/session_2026-11-25_13-56-31_204231_999999/sockets/plasma_store"
        validate_socket_filepath(socket)
        assert len(socket.encode()) <= 107
