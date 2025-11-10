import pytest
import json
import time
from unittest.mock import AsyncMock, Mock
from brain_service.services.memory_service import MemoryService


class TestMemoryService:
    """Test cases for MemoryService."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.set = AsyncMock(return_value=True)
        redis_mock.delete = AsyncMock(return_value=1)
        return redis_mock
    
    @pytest.fixture
    def memory_service(self, mock_redis):
        """Create MemoryService instance with mocked Redis."""
        return MemoryService(redis_client=mock_redis, ttl_sec=3600)
    
    @pytest.mark.asyncio
    async def test_get_stm_empty(self, memory_service, mock_redis):
        """Test getting STM when none exists."""
        result = await memory_service.get_stm("test_session")
        assert result is None
        mock_redis.get.assert_called_once_with("stm:test_session")
    
    @pytest.mark.asyncio
    async def test_get_stm_existing(self, memory_service, mock_redis):
        """Test getting existing STM."""
        stm_data = {
            "summary_v1": "Test summary",
            "salient_facts": ["fact1", "fact2"],
            "open_loops": ["question1"],
            "ts_updated": time.time()
        }
        mock_redis.get.return_value = json.dumps(stm_data)
        
        result = await memory_service.get_stm("test_session")
        assert result == stm_data
    
    @pytest.mark.asyncio
    async def test_update_stm_new(self, memory_service, mock_redis):
        """Test updating STM for new session."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        result = await memory_service.update_stm("test_session", messages)
        
        assert "summary_v1" in result
        assert "salient_facts" in result
        assert "open_loops" in result
        assert "ts_updated" in result
        
        # Verify Redis set was called
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "stm:test_session"
        assert call_args[1]["ex"] == 3600
    
    @pytest.mark.asyncio
    async def test_update_stm_merge_existing(self, memory_service, mock_redis):
        """Test updating STM with existing data."""
        existing_stm = {
            "summary_v1": "Old summary",
            "salient_facts": ["old_fact"],
            "open_loops": ["old_question"],
            "ts_updated": time.time() - 100
        }
        mock_redis.get.return_value = json.dumps(existing_stm)
        
        messages = [
            {"role": "user", "content": "New question?"},
            {"role": "assistant", "content": "New answer with new fact."}
        ]
        
        result = await memory_service.update_stm("test_session", messages)
        
        # Should merge existing and new data
        assert len(result["salient_facts"]) >= 1  # At least the old fact
        assert len(result["open_loops"]) >= 1     # At least the old question
    
    @pytest.mark.asyncio
    async def test_clear_stm(self, memory_service, mock_redis):
        """Test clearing STM."""
        result = await memory_service.clear_stm("test_session")
        assert result is True
        mock_redis.delete.assert_called_once_with("stm:test_session")
    
    def test_should_update_stm_by_messages(self, memory_service):
        """Test STM update trigger by message count."""
        # Should update when message count exceeds threshold
        assert memory_service.should_update_stm(10, 1000, trigger_msgs=8) is True
        assert memory_service.should_update_stm(5, 1000, trigger_msgs=8) is False
    
    def test_should_update_stm_by_tokens(self, memory_service):
        """Test STM update trigger by token count."""
        # Should update when token count exceeds threshold
        assert memory_service.should_update_stm(5, 3000, trigger_tokens=2000) is True
        assert memory_service.should_update_stm(5, 1000, trigger_tokens=2000) is False
    
    def test_simple_summary_generation(self, memory_service):
        """Test simple summary generation."""
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."}
        ]
        
        summary = memory_service._generate_simple_summary(messages)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "User" in summary or "Assistant" in summary
    
    def test_extract_salient_facts(self, memory_service):
        """Test salient facts extraction."""
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language that is widely used."}
        ]
        
        facts = memory_service._extract_salient_facts(messages)
        assert isinstance(facts, list)
        # Should extract the factual statement about Python
        assert len(facts) > 0
    
    def test_extract_open_loops(self, memory_service):
        """Test open loops extraction."""
        messages = [
            {"role": "user", "content": "What is Python? How does it work?"},
            {"role": "assistant", "content": "Python is a programming language, but I need to explain more."}
        ]
        
        loops = memory_service._extract_open_loops(messages)
        assert isinstance(loops, list)
        # Should extract the questions
        assert len(loops) > 0
    
    @pytest.mark.asyncio
    async def test_no_redis_client(self):
        """Test behavior when Redis client is None."""
        memory_service = MemoryService(redis_client=None, ttl_sec=3600)
        
        # Should return None for get_stm
        result = await memory_service.get_stm("test_session")
        assert result is None
        
        # Should return empty dict for update_stm
        result = await memory_service.update_stm("test_session", [])
        assert result == {}
        
        # Should return False for clear_stm
        result = await memory_service.clear_stm("test_session")
        assert result is False































