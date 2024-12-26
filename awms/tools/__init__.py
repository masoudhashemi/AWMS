from awms.tools.cot import CoTTool
from awms.tools.pal import PALTool

# Tool registry
TOOL_REGISTRY = {
    "CoT": CoTTool,
    "PAL": PALTool,
}


def get_tool(tool_name: str):
    tool_class = TOOL_REGISTRY.get(tool_name)
    if tool_class is not None:
        return tool_class()
    else:
        raise ValueError(f"Tool '{tool_name}' is not registered.")
