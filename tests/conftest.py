"""Keep regression tests offline unless an explicit HTTP mock is provided."""

import pytest
import requests


@pytest.fixture(autouse=True)
def block_unmocked_http(monkeypatch):
    """Fail if a UI rerun or test attempts an actual HTTP request."""
    def blocked(*args, **kwargs):
        raise AssertionError("Unexpected HTTP access in an offline regression test")
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
