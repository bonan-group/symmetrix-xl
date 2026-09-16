import os
from unittest.mock import patch

from symmetrix import Symmetrix


def debug_factorized_symmetrix(*args, **kwargs):
    """Construct a calculator with the debug-only generic factorized path."""

    kwargs["streamed_edges"] = "generic"
    with patch.dict(os.environ, {"SYMMETRIX_JIT_POLICY": "none"}):
        return Symmetrix(*args, **kwargs)
