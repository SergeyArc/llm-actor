from unittest.mock import MagicMock, patch
import pytest
from llm_actor.logger import BrokerLogger, _broker_log_record_patcher

def test_broker_log_record_patcher_basic():
    record = {"extra": {"actor_id": "actor-123", "pool_id": "pool-456"}}
    _broker_log_record_patcher(record)  # type: ignore
    
    assert record["extra"]["actor_tag"] == "[actor-123] "
    assert record["extra"]["pool_tag"] == "[pool pool-456] "
    assert record["extra"]["trace_tag"] == ""

def test_broker_log_record_patcher_empty():
    record = {"extra": {}}
    _broker_log_record_patcher(record)  # type: ignore
    
    assert record["extra"]["actor_tag"] == ""
    assert record["extra"]["pool_tag"] == ""
    assert record["extra"]["trace_tag"] == ""

def test_logger_get_logger_binds_name():
    with patch("llm_actor.logger.logger.bind") as mock_bind:
        BrokerLogger.get_logger("test-service")
        mock_bind.assert_called_with(name="test-service")

def test_logger_bind_context():
    with patch("llm_actor.logger.logger.bind") as mock_bind:
        BrokerLogger.bind_context(actor_id="a1", pool_id="p1")
        mock_bind.assert_called_with(actor_id="a1", pool_id="p1")

def test_setup_standard_logging():
    with patch("llm_actor.logger.logger.remove") as mock_remove:
        with patch("llm_actor.logger.logger.add") as mock_add:
            BrokerLogger.setup_standard_logging(level="DEBUG")
            assert mock_remove.called
            assert mock_add.called
            args, kwargs = mock_add.call_args
            assert kwargs["level"] == "DEBUG"
            assert "{extra[actor_tag]}" in kwargs["format"]
