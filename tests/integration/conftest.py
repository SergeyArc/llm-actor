
import pytest
from dotenv import load_dotenv

load_dotenv()

def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False, help="run integration tests"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        # Flag set: run integration tests
        return

    # No flag: skip tests under integration/
    skip_int = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip_int)

@pytest.fixture(scope="session")
def integration_enabled(request):
    return request.config.getoption("--integration")
