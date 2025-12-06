import pytest
from unittest.mock import AsyncMock
from brain_service.domain.chat.tools import ToolRegistry


class TestToolRegistry:
    """Test cases for ToolRegistry."""
    
    @pytest.fixture
    def tool_registry(self):
        """Create ToolRegistry instance."""
        return ToolRegistry()
    
    @pytest.fixture
    def mock_handler(self):
        """Mock async handler function."""
        return AsyncMock(return_value={"result": "success"})
    
    def test_register_tool(self, tool_registry, mock_handler):
        """Test tool registration."""
        schema = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool"
            }
        }
        
        tool_registry.register("test_tool", mock_handler, schema)
        
        # Check that tool was registered
        assert "test_tool" in tool_registry._map
        assert tool_registry._map["test_tool"] == mock_handler
        
        # Check that schema was added
        assert schema in tool_registry._schemas
    
    def test_get_tool_schemas(self, tool_registry, mock_handler):
        """Test getting tool schemas."""
        schema1 = {"type": "function", "function": {"name": "tool1"}}
        schema2 = {"type": "function", "function": {"name": "tool2"}}
        
        tool_registry.register("tool1", mock_handler, schema1)
        tool_registry.register("tool2", mock_handler, schema2)
        
        schemas = tool_registry.get_tool_schemas()
        assert len(schemas) == 2
        assert schema1 in schemas
        assert schema2 in schemas
    
    @pytest.mark.asyncio
    async def test_call_registered_tool(self, tool_registry, mock_handler):
        """Test calling a registered tool."""
        schema = {"type": "function", "function": {"name": "test_tool"}}
        tool_registry.register("test_tool", mock_handler, schema)
        
        result = await tool_registry.call("test_tool", arg1="value1", arg2="value2")
        
        assert result == {"result": "success"}
        mock_handler.assert_called_once_with(arg1="value1", arg2="value2")
    
    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, tool_registry):
        """Test calling an unknown tool."""
        result = await tool_registry.call("unknown_tool", arg1="value1")
        
        assert result == {"ok": False, "error": "unknown tool: unknown_tool"}
    
    @pytest.mark.asyncio
    async def test_call_tool_with_exception(self, tool_registry):
        """Test calling a tool that raises an exception."""
        async def failing_handler(**kwargs):
            raise ValueError("Test error")
        
        schema = {"type": "function", "function": {"name": "failing_tool"}}
        tool_registry.register("failing_tool", failing_handler, schema)
        
        result = await tool_registry.call("failing_tool", arg1="value1")
        
        assert result == {"ok": False, "error": "Test error"}
    
    def test_multiple_tools_registration(self, tool_registry):
        """Test registering multiple tools."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        schema1 = {"type": "function", "function": {"name": "tool1"}}
        schema2 = {"type": "function", "function": {"name": "tool2"}}
        
        tool_registry.register("tool1", handler1, schema1)
        tool_registry.register("tool2", handler2, schema2)
        
        assert len(tool_registry._map) == 2
        assert len(tool_registry._schemas) == 2
        assert "tool1" in tool_registry._map
        assert "tool2" in tool_registry._map
    
    @pytest.mark.asyncio
    async def test_tool_with_complex_arguments(self, tool_registry):
        """Test calling tool with complex arguments."""
        async def complex_handler(text: str, options: dict, count: int = 1):
            return {"processed": text, "options": options, "count": count}
        
        schema = {"type": "function", "function": {"name": "complex_tool"}}
        tool_registry.register("complex_tool", complex_handler, schema)
        
        result = await tool_registry.call(
            "complex_tool",
            text="Hello World",
            options={"format": "json", "pretty": True},
            count=5
        )
        
        assert result["processed"] == "Hello World"
        assert result["options"]["format"] == "json"
        assert result["count"] == 5




































