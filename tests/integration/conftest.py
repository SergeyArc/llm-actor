import pytest
import os
from dotenv import load_dotenv

load_dotenv()

def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False, help="run integration tests"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        # Если флаг передан, ничего не делаем, запускаем все
        return
    
    # Если флаг НЕ передан, помечаем все тесты в папке integration как пропущенные
    skip_int = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip_int)

@pytest.fixture(scope="session")
def integration_enabled(request):
    return request.config.getoption("--integration")
