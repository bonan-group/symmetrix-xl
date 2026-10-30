from __future__ import annotations

from types import SimpleNamespace


import conftest


class _CollectedItem:
    def __init__(self, *keywords: str):
        self.keywords = {keyword: True for keyword in keywords}
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


def test_backend_independent_collection_does_not_load_native_backend(monkeypatch):
    monkeypatch.delenv("SYMMETRIX_REQUIRED_TEST_MANIFEST", raising=False)

    def unexpected_query():
        raise AssertionError("backend-independent collection queried native state")

    monkeypatch.setattr(conftest, "_native_capabilities", unexpected_query)

    conftest.pytest_collection_modifyitems(
        SimpleNamespace(), [_CollectedItem("ordinary_test")]
    )


def test_oam_runtime_tests_are_skipped_on_the_unsupported_hip_backend(
    monkeypatch,
):
    monkeypatch.delenv("SYMMETRIX_REQUIRED_TEST_MANIFEST", raising=False)
    monkeypatch.setattr(
        conftest,
        "_native_capabilities",
        lambda: {"available": True, "backend": "hip"},
    )
    item = _CollectedItem("oam_runtime")

    conftest.pytest_collection_modifyitems(SimpleNamespace(), [item])

    assert len(item.markers) == 1
    assert item.markers[0].mark.name == "skip"
    assert item.markers[0].mark.kwargs == {
        "reason": "OAM execution is unsupported on the HIP backend"
    }


def test_oam_runtime_tests_remain_selected_on_supported_backends(
    monkeypatch,
):
    monkeypatch.delenv("SYMMETRIX_REQUIRED_TEST_MANIFEST", raising=False)
    for backend in ("host", "cuda"):
        monkeypatch.setattr(
            conftest,
            "_native_capabilities",
            lambda backend=backend: {"available": True, "backend": backend},
        )
        item = _CollectedItem("oam_runtime")

        conftest.pytest_collection_modifyitems(SimpleNamespace(), [item])

        assert item.markers == []


def test_host_tests_are_skipped_on_device_backends(monkeypatch):
    monkeypatch.delenv("SYMMETRIX_REQUIRED_TEST_MANIFEST", raising=False)
    monkeypatch.setattr(
        conftest,
        "_native_capabilities",
        lambda: {"available": True, "backend": "hip"},
    )
    item = _CollectedItem("host")

    conftest.pytest_collection_modifyitems(SimpleNamespace(), [item])

    assert len(item.markers) == 1
    assert item.markers[0].mark.name == "skip"
    assert item.markers[0].mark.kwargs == {"reason": "requires a host backend"}
