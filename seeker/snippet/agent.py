#date: 2025-11-17T16:45:18Z
#url: https://api.github.com/gists/0eac686edf5105699f9429ce190967b0
#owner: https://api.github.com/users/AlyceOsbourne

"""
A simplified AI agent with tool calling and automatic schema generation.
"""

import openai
import inspect
import json
from typing import Annotated, get_origin, get_args, Any, Callable
from dataclasses import dataclass, field


@dataclass
class Tool:
    """Represents a tool function with automatic schema generation."""
    
    callable: Callable[..., Any]
    schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Generate schema from the callable if not already set."""
        if self.schema is None:
            self.schema = self._generate_schema()

    @staticmethod
    def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
        """Convert a type annotation to a JSON schema."""
        schema: dict[str, Any] = {"type": "string"}
        description: str | None = None
        origin = get_origin(annotation)

        if origin is Annotated:
            base_type, *meta = get_args(annotation)
            schema = Tool._annotation_to_schema(base_type)
            if meta:
                description = str(meta[0])
        elif annotation in [int, float]:
            schema = {"type": "number"}
        elif annotation is bool:
            schema = {"type": "boolean"}
        elif annotation is str:
            schema = {"type": "string"}
        elif annotation == dict:
            schema = {"type": "object"}
        elif annotation == list:
            schema = {"type": "array"}

        if description:
            schema["description"] = description

        return schema

    def _generate_schema(self) -> dict[str, Any]:
        """Generate the OpenAI tool schema from the callable's signature."""
        sig = inspect.signature(self.callable)

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for name, param in sig.parameters.items():
            annotation = param.annotation
            schema = self._annotation_to_schema(annotation)
            parameters["properties"][name] = schema
            if param.default is param.empty:
                parameters["required"].append(name)

        return {
            "type": "function",
            "function": {
                "name": self.callable.__name__,
                "description": self.callable.__doc__ or "No description provided.",
                "parameters": parameters,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make Tool callable, delegating to the underlying callable."""
        return self.callable(*args, **kwargs)


@dataclass
class Agent:
    """A simple AI agent with tool calling support."""
    
    model: str = "pyxie"
    system_prompt: str = "You are a helpful assistant."
    base_url: str | None = "http://127.0.0.1:1234/v1"
    api_key: str | None = "NO_API_KEY"
    tools: dict[str, Tool] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    client: Any = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the OpenAI client."""
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        if self.messages and self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})
        elif not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a function as a tool."""
        tool_obj = Tool(callable=func)
        self.tools[func.__name__] = tool_obj
        return func

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get the list of tool schemas for API calls."""
        return [tool.schema for tool in self.tools.values() if tool.schema is not None]

    def _execute_tool(self, tool_call: Any) -> dict[str, Any]:
        """Execute a tool call and return the result."""
        fn_name = tool_call.function.name
        tool = self.tools.get(fn_name)
        
        if not tool:
            return {"error": f"Tool '{fn_name}' not found"}

        try:
            args = json.loads(tool_call.function.arguments or "{}")
            result = tool(**args)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant's response."""
        self.messages.append({"role": "user", "content": user_message})

        while True:
            api_kwargs = {
                "model": self.model,
                "messages": self.messages,
            }

            tool_schemas = self._get_tool_schemas()
            if tool_schemas:
                api_kwargs["tools"] = tool_schemas

            response = self.client.chat.completions.create(**api_kwargs)
            message = response.choices[0].message

            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (message.tool_calls or [])
                ],
            })

            if not message.tool_calls:
                return message.content or ""

            for tool_call in message.tool_calls:
                result = self._execute_tool(tool_call)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })


if __name__ == "__main__":
    agent = Agent(
        model="pyxie",
        system_prompt="You are a helpful assistant that can perform calculations.",
    )

    @agent.tool
    def add(a: Annotated[int, "First number"], b: Annotated[int, "Second number"]) -> dict[str, int]:
        """Add two numbers together."""
        return {"result": a + b}

    @agent.tool
    def multiply(
        a: Annotated[int, "First number"],
        b: Annotated[int, "Second number"],
    ) -> dict[str, int]:
        """Multiply two numbers together."""
        return {"result": a * b}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        print(agent.chat(user_input))
