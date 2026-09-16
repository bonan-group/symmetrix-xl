import sys

import pytest
import symmetrix


def test_team_scratch_admission_accepts_small_request_and_rejects_limit():
    admit = getattr(symmetrix, "_team_scratch_admission_for_testing", None)
    if admit is None:
        pytest.skip("installed native extension predates team scratch admission")

    assert admit(17, 8) == 136
    with pytest.raises(
        ValueError,
        match=r"team-scratch-test requires .* available limit is .* bytes",
    ):
        admit(sys.maxsize, 1)


def test_team_scratch_admission_rejects_size_overflow():
    admit = getattr(symmetrix, "_team_scratch_admission_for_testing", None)
    if admit is None:
        pytest.skip("installed native extension predates team scratch admission")

    with pytest.raises(ValueError, match="team scratch size overflow"):
        admit(sys.maxsize, 3)
