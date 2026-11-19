import pytest

from injbench.world import build_world


@pytest.fixture(scope="session")
def world():
    return build_world(0)
