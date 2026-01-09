"""BrowserUse CUA Client for Stagehand.

This client uses the browser-use cloud API (ChatBrowserUse) as the model provider
for computer use agent tasks, with DOM extraction for optimal model accuracy.
"""

import asyncio
import base64
import io
import json
import os
import random
import uuid
from typing import Any, Optional, Union

import httpx
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..dom.service import DomService
from ..dom.views import DEFAULT_INCLUDE_ATTRIBUTES, DOMState
from ..handlers.cua_handler import CUAHandler, StagehandFunctionName
from ..types.agent import (
    ActionExecutionResult,
    AgentAction,
    AgentActionType,
    AgentConfig,
    AgentExecuteOptions,
    AgentResult,
    AgentUsage,
    FunctionArguments,
    Point,
)
from .client import AgentClient


# =============================================================================
# Action Parameter Models (matching browser-use SDK's tools/views.py)
# =============================================================================

class ClickElementAction(BaseModel):
    """Click an element by index or coordinates."""
    index: Optional[int] = Field(default=None, ge=1, description="Element index from browser_state")
    coordinate_x: Optional[int] = Field(default=None, description="X coordinate relative to viewport")
    coordinate_y: Optional[int] = Field(default=None, description="Y coordinate relative to viewport")


class InputTextAction(BaseModel):
    """Type text into an input field."""
    index: int = Field(ge=0, description="Element index from browser_state")
    text: str = Field(description="Text to type")
    clear: bool = Field(default=True, description="Clear existing text before typing")


class ScrollAction(BaseModel):
    """Scroll the page."""
    down: bool = Field(default=True, description="True to scroll down, False to scroll up")
    pages: float = Field(default=1.0, description="Number of pages to scroll (0.5=half, 1=full, 10=to end)")
    index: Optional[int] = Field(default=None, description="Optional element index to scroll within")


class NavigateAction(BaseModel):
    """Navigate to a URL."""
    url: str = Field(description="URL to navigate to")
    new_tab: bool = Field(default=False, description="Open in new tab")


class GoBackAction(BaseModel):
    """Go back in browser history."""
    description: Optional[str] = Field(default=None, description="Optional description")


class SendKeysAction(BaseModel):
    """Send keyboard keys."""
    keys: str = Field(description="Keys to send (e.g., 'Enter', 'Escape', 'Control+c')")


class DoneAction(BaseModel):
    """Mark task as complete."""
    text: str = Field(description="Final message describing the result")
    success: bool = Field(default=True, description="Whether task completed successfully")


class SearchAction(BaseModel):
    """Search using a search engine."""
    query: str = Field(description="Search query")
    engine: str = Field(default="duckduckgo", description="Search engine to use")


class WaitAction(BaseModel):
    """Wait for a specified time."""
    seconds: int = Field(default=3, description="Seconds to wait")


class SwitchTabAction(BaseModel):
    """Switch to a different tab."""
    tab_id: str = Field(description="Tab ID to switch to")


class CloseTabAction(BaseModel):
    """Close a tab."""
    tab_id: str = Field(description="Tab ID to close")


class ExtractAction(BaseModel):
    """Extract information from the page."""
    query: str = Field(description="What to extract")


# =============================================================================
# Action Model - Union of all action types (like browser-use's dynamic model)
# =============================================================================

class BrowserUseActionModel(BaseModel):
    """Action model with one action field set at a time."""
    model_config = ConfigDict(extra="forbid")

    # Use simple Optional types without Field() for cleaner schema
    click_element: Optional[ClickElementAction] = None
    input_text: Optional[InputTextAction] = None
    scroll: Optional[ScrollAction] = None
    navigate: Optional[NavigateAction] = None
    go_back: Optional[GoBackAction] = None
    send_keys: Optional[SendKeysAction] = None
    done: Optional[DoneAction] = None
    search: Optional[SearchAction] = None
    wait: Optional[WaitAction] = None
    switch_tab: Optional[SwitchTabAction] = None
    close_tab: Optional[CloseTabAction] = None
    extract: Optional[ExtractAction] = None

    def get_action(self) -> tuple[str, BaseModel] | None:
        """Get the set action type and its parameters."""
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if value is not None:
                return (field_name, value)
        return None


# =============================================================================
# Agent Output Model (matching browser-use SDK's agent/views.py)
# =============================================================================

class BrowserUseAgentOutput(BaseModel):
    """Output format for browser-use API responses."""
    model_config = ConfigDict(extra="forbid")

    # Use simple Optional types without Field() for cleaner schema
    thinking: Optional[str] = None
    evaluation_previous_goal: Optional[str] = None
    memory: Optional[str] = None
    next_goal: Optional[str] = None
    action: list[BrowserUseActionModel] = Field(
        ...,
        json_schema_extra={"min_items": 1},
    )

    @classmethod
    def model_json_schema(cls, **kwargs):
        """Override to set required fields like browser-use does."""
        schema = super().model_json_schema(**kwargs)
        schema["required"] = ["evaluation_previous_goal", "memory", "next_goal", "action"]
        # Match browser-use's naming convention
        schema["title"] = "AgentOutput"
        if "$defs" in schema and "BrowserUseActionModel" in schema["$defs"]:
            schema["$defs"]["ActionModel"] = schema["$defs"].pop("BrowserUseActionModel")
            # Update references
            if "action" in schema["properties"]:
                action_prop = schema["properties"]["action"]
                if "items" in action_prop and "$ref" in action_prop["items"]:
                    action_prop["items"]["$ref"] = "#/$defs/ActionModel"
        return schema

# HTTP status codes that should trigger a retry
# Note: 400 is included because browser-use API sometimes returns transient 400 errors
RETRYABLE_STATUS_CODES = {400, 429, 500, 502, 503, 504}

# Browser-use API endpoint
DEFAULT_BASE_URL = "https://llm.api.browser-use.com"

# System prompt for browser-use model
SYSTEM_PROMPT = """You are a browser automation agent. You can interact with web pages by clicking elements, typing text, scrolling, and navigating.

When you see <browser_state>, it contains the interactive elements on the page with their indices in brackets like [1], [2], etc.
Use these indices to interact with elements.

Available actions:
- click_element: Click on an element by index or coordinates
- input_text: Type text into an input field
- scroll: Scroll the page
- navigate: Go to a URL
- go_back: Go back in browser history
- send_keys: Send keyboard keys
- done: Complete the task with a result

Always respond with valid JSON containing your thinking, memory, next_goal, and action list."""


class BrowserUseCUAClient(AgentClient):
    """Client for browser-use cloud API with DOM extraction."""

    # Key mapping for browser-use to Playwright
    KEY_MAPPING = {
        "enter": "Enter",
        "return": "Enter",
        "esc": "Escape",
        "escape": "Escape",
        "tab": "Tab",
        "backspace": "Backspace",
        "delete": "Delete",
        "space": " ",
        "arrowup": "ArrowUp",
        "arrowdown": "ArrowDown",
        "arrowleft": "ArrowLeft",
        "arrowright": "ArrowRight",
        "pageup": "PageUp",
        "pagedown": "PageDown",
        "home": "Home",
        "end": "End",
    }

    # Screenshot resize dimensions (matches browser-use SDK for Claude models)
    LLM_SCREENSHOT_SIZE = (1400, 850)

    def __init__(
        self,
        model: str,
        instructions: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        logger: Optional[Any] = None,
        handler: Optional[CUAHandler] = None,
        viewport: Optional[dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(model, instructions, config, logger, handler)

        # API configuration - accept both apiKey (camelCase) and api_key (snake_case)
        self.api_key = None
        if config and config.options:
            self.api_key = config.options.get("apiKey") or config.options.get("api_key")
        if not self.api_key:
            self.api_key = os.getenv("BROWSER_USE_API_KEY")

        self.base_url = (
            config.options.get("baseUrl") if config and config.options else None
        ) or os.getenv("BROWSER_USE_LLM_URL", DEFAULT_BASE_URL)

        if not self.api_key:
            raise ValueError(
                "BROWSER_USE_API_KEY is required. "
                "Get your key at https://cloud.browser-use.com/new-api-key"
            )

        # Normalize model name
        if model == "bu-latest":
            self.model = "bu-1-0"
        else:
            self.model = model

        # Viewport
        self.viewport = viewport or {"width": 1288, "height": 711}

        # DOM service (initialized lazily)
        self._dom_service: Optional[DomService] = None
        self._current_dom_state: Optional[DOMState] = None

        # Request configuration
        self.timeout = kwargs.get("timeout", 120.0)
        # Browser-use API can be flaky with transient 400 errors, so use higher retry count
        self.max_retries = kwargs.get("max_retries", 10)

        # Session ID for sticky routing (same session → same container)
        # This helps with API stability by routing all requests to the same backend
        self.session_id = str(uuid.uuid4())

        self.logger.info(
            f"BrowserUseCUAClient initialized for model: {self.model} (session: {self.session_id[:8]}...)",
            category=StagehandFunctionName.AGENT,
        )

    async def _get_dom_service(self) -> DomService:
        """Get or create DOM service."""
        if self._dom_service is None:
            self._dom_service = DomService(
                page=self.handler.page,
                logger=self.logger,
            )
        return self._dom_service

    async def run_task(
        self,
        instruction: str,
        max_steps: int = 20,
        options: Optional[AgentExecuteOptions] = None,
    ) -> AgentResult:
        """Run a browser automation task using browser-use API."""
        if self.config and self.config.max_steps is not None:
            max_steps = self.config.max_steps

        self.logger.debug(
            f"BrowserUse CUA starting task: '{instruction}' with max_steps: {max_steps}",
            category=StagehandFunctionName.AGENT,
        )

        if not self.handler:
            self.logger.error(
                "CUAHandler not available for BrowserUseCUAClient.",
                category=StagehandFunctionName.AGENT,
            )
            return AgentResult(
                completed=False,
                actions=[],
                message="Internal error: Handler not set.",
                usage=AgentUsage(input_tokens=0, output_tokens=0, inference_time_ms=0),
            )

        # Inject cursor for visual feedback
        await self.handler.inject_cursor()

        # Get initial state
        current_screenshot_b64 = await self.handler.get_screenshot_base64()
        dom_service = await self._get_dom_service()
        self._current_dom_state = await dom_service.get_dom_state()

        # Format initial messages
        messages = self._format_initial_messages(
            instruction, current_screenshot_b64
        )

        actions_taken: list[AgentAction] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_inference_time_ms = 0
        final_message: Optional[str] = None
        task_completed = False

        for step_count in range(max_steps):
            self.logger.info(
                f"BrowserUse CUA - Step {step_count + 1}/{max_steps}",
                category=StagehandFunctionName.AGENT,
            )

            start_time = asyncio.get_event_loop().time()
            try:
                response = await self._make_api_call(messages)
                end_time = asyncio.get_event_loop().time()
                total_inference_time_ms += int((end_time - start_time) * 1000)

                # Extract usage - with safety check
                if isinstance(response, dict):
                    usage = response.get("usage", {})
                    total_input_tokens += usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                    total_output_tokens += usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

            except Exception as e:
                self.logger.error(
                    f"BrowserUse API call failed: {e}",
                    category=StagehandFunctionName.AGENT,
                )
                return AgentResult(
                    actions=[act.action for act in actions_taken if act.action],
                    message=f"API error: {e}",
                    completed=False,
                    usage=AgentUsage(
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        inference_time_ms=total_inference_time_ms,
                    ),
                )

            # Process response
            (
                agent_actions,
                reasoning,
                is_done,
                done_message,
            ) = await self._process_provider_response(response)

            if reasoning:
                self.logger.info(
                    f"Model reasoning: {reasoning}",
                    category=StagehandFunctionName.AGENT,
                )
                final_message = reasoning

            if is_done:
                task_completed = True
                if done_message:
                    final_message = done_message
                self.logger.info(
                    f"Task completed. Message: {final_message}",
                    category=StagehandFunctionName.AGENT,
                )
                break

            # Execute actions
            if agent_actions:
                for idx, agent_action in enumerate(agent_actions):
                    actions_taken.append(agent_action)

                    # Execute the action
                    action_result: dict[str, Any] = (
                        await self.handler.perform_action(agent_action)
                    )

                    # Get new state after action
                    current_screenshot_b64 = await self.handler.get_screenshot_base64()
                    self._current_dom_state = await dom_service.get_dom_state()

                    # Format feedback
                    feedback = self._format_action_feedback(
                        action=agent_action,
                        action_result=action_result,
                        new_screenshot_base64=current_screenshot_b64,
                    )
                    messages.extend(feedback)

            else:
                # No actions returned - continue loop, only exit on explicit done
                self.logger.debug(
                    "Model did not return any actions. Continuing to wait for done signal.",
                    category=StagehandFunctionName.AGENT,
                )

        return AgentResult(
            actions=[act.action for act in actions_taken if act.action],
            message=final_message or "Max steps reached.",
            completed=task_completed,
            usage=AgentUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                inference_time_ms=total_inference_time_ms,
            ),
        )

    def _format_initial_messages(
        self, instruction: str, screenshot_base64: Optional[str]
    ) -> list[dict[str, Any]]:
        """Format initial messages for browser-use API."""
        messages = []

        # System message
        system_content = self.instructions or SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_content})

        # User message with task + DOM state + screenshot
        user_content = []

        # Add task instruction
        user_content.append({"type": "text", "text": f"Task: {instruction}"})

        # Add DOM state
        if self._current_dom_state:
            dom_text = self._current_dom_state.serialize(DEFAULT_INCLUDE_ATTRIBUTES)
            user_content.append({
                "type": "text",
                "text": f"<browser_state>\n{dom_text}\n</browser_state>",
            })

        # Add current URL
        if self._current_dom_state and self._current_dom_state.url:
            user_content.append({
                "type": "text",
                "text": f"Current URL: {self._current_dom_state.url}",
            })

        # Add screenshot (resized/compressed to reduce payload size)
        if screenshot_base64:
            resized_screenshot = self._resize_screenshot(screenshot_base64)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{resized_screenshot}",
                    "media_type": "image/jpeg",
                    "detail": "auto",
                },
            })

        messages.append({"role": "user", "content": user_content})

        return messages

    async def _process_provider_response(
        self, response: dict[str, Any]
    ) -> tuple[list[AgentAction], Optional[str], bool, Optional[str]]:
        """Process browser-use API response.

        With output_format, the response is structured:
        {
            'completion': {
                'thinking': '...',
                'memory': '...',
                'next_goal': '...',
                'action': [{'click_element': {'index': 2}}]
            },
            'usage': {...},
            'cost': {...}
        }

        Without output_format (fallback), completion is a string with XML tags.

        Returns:
            - List of AgentActions to execute
            - Reasoning text
            - Whether task is done
            - Done message (if task is done)
        """
        if not isinstance(response, dict):
            return [], None, False, None

        completion = response.get("completion", {})

        # Handle structured dict response (when output_format is provided)
        if isinstance(completion, dict):
            return await self._process_structured_completion(completion)

        # Fallback: Handle string response with XML tags (when output_format is not used)
        if isinstance(completion, str):
            return await self._process_string_completion(completion)

        return [], None, False, None

    async def _process_structured_completion(
        self, completion: dict[str, Any]
    ) -> tuple[list[AgentAction], Optional[str], bool, Optional[str]]:
        """Process structured completion dict from browser-use API.

        Uses Pydantic validation to ensure the response matches our schema.
        """
        # Try to validate with Pydantic model
        try:
            validated = BrowserUseAgentOutput.model_validate(completion)
        except Exception:
            # Fall back to dict-based parsing
            return await self._process_structured_completion_dict(completion)

        # Extract reasoning from validated model
        reasoning_parts = []
        if validated.thinking:
            reasoning_parts.append(f"Thinking: {validated.thinking}")
        if validated.next_goal:
            reasoning_parts.append(f"Goal: {validated.next_goal}")
        if validated.memory:
            reasoning_parts.append(f"Memory: {validated.memory}")
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else None

        agent_actions = []
        is_done = False
        done_message = None

        for action_model in validated.action:
            # Get the action type and params from the model
            action_info = action_model.get_action()
            if not action_info:
                continue

            action_type, action_params = action_info

            # Check for done action
            if action_type == "done":
                is_done = True
                done_message = action_params.text if hasattr(action_params, 'text') else "Task completed"
                continue

            # Convert to AgentAction using the validated model
            agent_action = await self._convert_action_model(action_type, action_params)
            if agent_action:
                agent_actions.append(agent_action)

        return agent_actions, reasoning, is_done, done_message

    async def _process_structured_completion_dict(
        self, completion: dict[str, Any]
    ) -> tuple[list[AgentAction], Optional[str], bool, Optional[str]]:
        """Fallback: Process structured completion as raw dict (without Pydantic validation)."""
        thinking = completion.get("thinking")
        memory = completion.get("memory")
        next_goal = completion.get("next_goal")

        reasoning_parts = []
        if thinking:
            reasoning_parts.append(f"Thinking: {thinking}")
        if next_goal:
            reasoning_parts.append(f"Goal: {next_goal}")
        if memory:
            reasoning_parts.append(f"Memory: {memory}")
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else None

        action_list = completion.get("action", [])
        if not isinstance(action_list, list):
            action_list = [action_list] if action_list else []

        agent_actions = []
        is_done = False
        done_message = None

        for action_data in action_list:
            if not isinstance(action_data, dict):
                continue

            # Check for done action
            if "done" in action_data:
                is_done = True
                done_info = action_data["done"]
                if isinstance(done_info, dict):
                    done_message = done_info.get("text", done_info.get("message", "Task completed"))
                else:
                    done_message = str(done_info) if done_info else "Task completed"
                continue

            # Convert to AgentAction
            agent_action = await self._convert_action(action_data)
            if agent_action:
                agent_actions.append(agent_action)

        return agent_actions, reasoning, is_done, done_message

    async def _process_string_completion(
        self, completion: str
    ) -> tuple[list[AgentAction], Optional[str], bool, Optional[str]]:
        """Process string completion with XML action tags (fallback mode)."""
        # Parse the completion string to extract reasoning and actions
        reasoning, action_dicts = self._parse_completion_string(completion)

        agent_actions = []
        is_done = False
        done_message = None

        for action_data in action_dicts:
            if action_data is None:
                continue

            # Check for done action
            if "done" in action_data:
                is_done = True
                done_message = action_data["done"].get("text", "Task completed")
                continue

            # Convert to AgentAction
            agent_action = await self._convert_action(action_data)
            if agent_action:
                agent_actions.append(agent_action)

        return agent_actions, reasoning, is_done, done_message

    def _parse_completion_string(self, completion: str) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Parse the completion string to extract reasoning and actions.

        Handles multiple action formats:
        1. Self-closing XML: <action type="click" selector="[2]"/>
        2. Content-based: <action>click [2]</action>
        3. Wrapper tags: <action_sequence>, <action_set>, <execute_action>, etc.
        4. No tags: Action: type_text(2, "text") or just type [2] text

        Args:
            completion: String like 'Reasoning...\n\n<action type="click" selector="[2]"/>'

        Returns:
            - Reasoning text (everything before first <action> tag)
            - List of parsed action dicts
        """
        import re

        actions = []

        # Pattern 1: Self-closing XML tags like <action type="click" selector="[2]"/>
        # This matches <action followed by attributes and ending with />
        self_closing_pattern = r'<action\s+([^>]*?)\s*/>'
        self_closing_matches = re.finditer(self_closing_pattern, completion, re.DOTALL)

        for match in self_closing_matches:
            attrs_str = match.group(1)
            action_dict = self._parse_action_attributes(attrs_str)
            if action_dict:
                actions.append(action_dict)

        # Pattern 2: Content-based tags - match various tag names used by the model
        # Matches: <action>, <action_sequence>, <action_set>, <execute_action>, etc.
        content_pattern = r'<(action|action_sequence|action_set|execute_action)>(.*?)</\1>'
        content_matches = re.findall(content_pattern, completion, re.DOTALL)

        for tag_name, content in content_matches:
            # Check if content contains nested <action> tags
            nested_actions = re.findall(r'<action>(.*?)</action>', content, re.DOTALL)
            if nested_actions:
                for nested_content in nested_actions:
                    parsed_actions = self._parse_action_content(nested_content.strip())
                    if parsed_actions:
                        if isinstance(parsed_actions, list):
                            actions.extend(parsed_actions)
                        else:
                            actions.append(parsed_actions)
            else:
                # Content might have multiple actions, try to parse each
                parsed_actions = self._parse_action_content(content.strip())
                if parsed_actions:
                    if isinstance(parsed_actions, list):
                        actions.extend(parsed_actions)
                    else:
                        actions.append(parsed_actions)

        # Pattern 2.5: Standalone action tags like <scroll_to_bottom />, <go_back />, etc.
        # These are self-closing tags without attributes
        standalone_self_closing = re.findall(
            r'<(scroll_to_bottom|scroll_to_top|go_back|refresh|wait)\s*/>',
            completion, re.IGNORECASE
        )
        for tag_name in standalone_self_closing:
            tag_lower = tag_name.lower()
            if tag_lower == "scroll_to_bottom":
                actions.append({"scroll": {"direction": "down", "amount": "page"}})
            elif tag_lower == "scroll_to_top":
                actions.append({"scroll": {"direction": "up", "amount": "page"}})
            elif tag_lower == "go_back":
                actions.append({"go_back": {}})
            elif tag_lower == "refresh":
                actions.append({"navigate": {"url": "reload"}})
            elif tag_lower == "wait":
                actions.append({"wait": {"seconds": 2}})

        # Pattern 2.6: Standalone action tags with content like <scroll_to_bottom></scroll_to_bottom>
        standalone_content = re.findall(
            r'<(scroll_to_bottom|scroll_to_top|go_back|refresh|wait)>\s*</\1>',
            completion, re.IGNORECASE
        )
        for tag_name in standalone_content:
            tag_lower = tag_name.lower()
            if tag_lower == "scroll_to_bottom":
                actions.append({"scroll": {"direction": "down", "amount": "page"}})
            elif tag_lower == "scroll_to_top":
                actions.append({"scroll": {"direction": "up", "amount": "page"}})
            elif tag_lower == "go_back":
                actions.append({"go_back": {}})
            elif tag_lower == "refresh":
                actions.append({"navigate": {"url": "reload"}})
            elif tag_lower == "wait":
                actions.append({"wait": {"seconds": 2}})

        # Pattern 2.7: Execute script tags <execute_script>...</execute_script>
        script_matches = re.findall(
            r'<execute_script>\s*(.*?)\s*</execute_script>',
            completion, re.DOTALL | re.IGNORECASE
        )
        for script in script_matches:
            script = script.strip()
            if script:
                actions.append({"execute_script": {"script": script}})

        # Pattern 3: No tags - try to parse the whole completion as an action
        # Look for patterns like "Action: type_text(...)" or "type [2] text"
        if not actions:
            # Try parsing the completion directly (strip "Action:" prefix if present)
            action_str = completion.strip()
            if action_str.lower().startswith("action:"):
                action_str = action_str[7:].strip()

            # First try parsing line by line for multi-line action sequences
            # e.g., "Type X into Y\nClick button Z"
            lines = [line.strip() for line in action_str.split('\n') if line.strip()]
            if len(lines) > 1:
                for line in lines:
                    # Skip lines that look like explanatory text (too long, no action keywords)
                    if len(line) > 200:
                        continue
                    # Check if line starts with an action keyword
                    action_keywords = ['type', 'click', 'input', 'scroll', 'navigate', 'goto', 'press', 'send_keys', 'wait', 'done', 'go_back']
                    if any(line.lower().startswith(kw) for kw in action_keywords):
                        parsed = self._parse_action_content(line)
                        if parsed:
                            if isinstance(parsed, list):
                                actions.extend(parsed)
                            else:
                                actions.append(parsed)

            # If no actions from line-by-line, try parsing the whole string
            if not actions:
                parsed_actions = self._parse_action_content(action_str)
                if parsed_actions:
                    if isinstance(parsed_actions, list):
                        actions.extend(parsed_actions)
                    else:
                        actions.append(parsed_actions)

        # Extract reasoning (everything before the first <action tag)
        first_action_pos = completion.find("<action")
        if first_action_pos > 0:
            reasoning = completion[:first_action_pos].strip()
        elif first_action_pos == 0:
            reasoning = None
        else:
            # No action tags found - if we parsed actions, reasoning is None
            reasoning = None if actions else (completion.strip() if completion.strip() else None)

        return reasoning, actions

    def _sanitize_text_value(self, text: str) -> str:
        """Sanitize text value by removing XML/markup artifacts and formatting.

        The model sometimes outputs malformed responses where XML tags leak into
        text values, e.g., 'email@example.com"\n</execute_action>'. It also may
        include markdown formatting like backticks around values. This method
        cleans such artifacts.

        Args:
            text: The raw text value to sanitize

        Returns:
            Cleaned text with artifacts removed
        """
        import re

        if not text:
            return text

        # Remove trailing XML-like tags (e.g., </execute_action>, </action>, etc.)
        # These can appear when the model's output is malformed
        text = re.sub(r'</?[\w_-]+>.*$', '', text, flags=re.DOTALL)

        # Remove trailing escaped newlines and quotes that precede XML tags
        # Pattern: trailing quote + optional whitespace/newlines + end
        text = re.sub(r'["\']?\s*\\n\s*$', '', text)
        text = re.sub(r'["\']?\s*$', '', text)

        # Strip any remaining leading/trailing whitespace
        text = text.strip()

        # Strip surrounding backticks (markdown inline code formatting)
        # The model sometimes outputs text like `email@example.com` with backticks
        if text.startswith('`') and text.endswith('`'):
            text = text[1:-1]

        # Also strip surrounding quotes if present (in case of double-quoting)
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]

        return text

    def _parse_action_attributes(self, attrs_str: str) -> Optional[dict[str, Any]]:
        """Parse action attributes from a self-closing tag.

        Example: type="click" selector="[2]"

        Args:
            attrs_str: The attributes string from inside the tag

        Returns:
            Action data dict or None if parsing fails
        """
        import re

        # Extract all attribute key="value" pairs
        # Handle double-quoted and single-quoted attributes separately
        # to properly support nested quotes like selector="[id='foo']"
        double_quoted = re.findall(r'(\w+)\s*=\s*"([^"]*)"', attrs_str)
        single_quoted = re.findall(r"(\w+)\s*=\s*'([^']*)'", attrs_str)
        attrs = dict(double_quoted)
        attrs.update(dict(single_quoted))

        action_type = attrs.get("type", "").lower()

        # Helper to extract index from selector, index, or target attribute
        def get_index_from_attrs(attrs: dict) -> Optional[int]:
            # First check for explicit index attribute
            if "index" in attrs:
                try:
                    return int(attrs["index"])
                except ValueError:
                    pass

            # Check for target attribute (model sometimes uses target instead of index)
            if "target" in attrs:
                target = attrs["target"]
                # Target might be just a number or [number]
                target_match = re.search(r'^\[?(\d+)\]?$', target)
                if target_match:
                    return int(target_match.group(1))

            # Then check selector for [digit] pattern (e.g., "[2]" or "2")
            selector = attrs.get("selector", "")

            # Try to match [digit] pattern
            index_match = re.search(r'^\[(\d+)\]$', selector)
            if index_match:
                return int(index_match.group(1))

            # Maybe selector is just a number
            if selector.isdigit():
                return int(selector)

            return None

        if action_type == "click":
            index = get_index_from_attrs(attrs)
            selector = attrs.get("selector", "")

            if index is not None:
                return {"click_element": {"index": index}}
            elif selector:
                # CSS selector - pass it through for handling
                return {"click_element": {"css_selector": selector}}

        elif action_type in ("input_text", "type", "input"):
            index = get_index_from_attrs(attrs)
            selector = attrs.get("selector", "")
            text = self._sanitize_text_value(attrs.get("text", attrs.get("value", "")))

            if index is not None:
                return {"input_text": {"index": index, "text": text}}
            elif selector:
                # CSS selector - pass it through for handling
                return {"input_text": {"css_selector": selector, "text": text}}

        elif action_type == "scroll":
            direction = attrs.get("direction", "down").lower()
            amount = attrs.get("amount", "1")
            try:
                pages = int(amount)
            except ValueError:
                pages = 1
            return {"scroll": {"down": direction == "down", "pages": pages}}

        elif action_type == "done":
            message = attrs.get("message", attrs.get("text", "Task completed"))
            return {"done": {"text": message}}

        elif action_type in ("navigate", "goto"):
            url = attrs.get("url", attrs.get("href", ""))
            url = self._sanitize_text_value(url)
            return {"navigate": {"url": url}}

        elif action_type in ("go_back", "back"):
            return {"go_back": {}}

        elif action_type in ("send_keys", "press", "key"):
            keys = attrs.get("keys", attrs.get("key", ""))
            return {"send_keys": {"keys": keys}}

        elif action_type == "wait":
            seconds = attrs.get("seconds", attrs.get("time", "3"))
            try:
                seconds = int(seconds)
            except ValueError:
                seconds = 3
            return {"wait": {"seconds": seconds}}

        elif action_type == "search":
            query = attrs.get("query", "")
            engine = attrs.get("engine", "duckduckgo")
            return {"search": {"query": query, "engine": engine}}

        elif action_type in ("switch", "switch_tab"):
            tab_id = attrs.get("tab_id", attrs.get("tab", ""))
            return {"switch_tab": {"tab_id": tab_id}}

        elif action_type in ("close", "close_tab"):
            tab_id = attrs.get("tab_id", attrs.get("tab", ""))
            return {"close_tab": {"tab_id": tab_id}}

        elif action_type == "extract":
            query = attrs.get("query", "")
            return {"extract": {"query": query}}

        elif action_type == "find_text":
            text = attrs.get("text", "")
            return {"find_text": {"text": text}}

        elif action_type == "screenshot":
            return {"screenshot": {}}

        elif action_type == "upload_file":
            selector = attrs.get("selector", "")
            path = attrs.get("path", attrs.get("file", ""))
            index_match = re.search(r'\[(\d+)\]', selector)
            if index_match:
                index = int(index_match.group(1))
                return {"upload_file": {"index": index, "path": path}}

        elif action_type == "evaluate":
            code = attrs.get("code", "")
            return {"evaluate": {"code": code}}

        elif action_type == "select_dropdown":
            selector = attrs.get("selector", "")
            text = attrs.get("text", attrs.get("option", ""))
            index_match = re.search(r'\[(\d+)\]', selector)
            if index_match:
                index = int(index_match.group(1))
                return {"select_dropdown": {"index": index, "text": text}}

        elif action_type == "dropdown_options":
            selector = attrs.get("selector", "")
            index_match = re.search(r'\[(\d+)\]', selector)
            if index_match:
                index = int(index_match.group(1))
                return {"dropdown_options": {"index": index}}

        return None

    def _parse_action_content(self, action_str: str) -> Optional[dict[str, Any] | list[dict[str, Any]]]:
        """Parse an action string from content-based tags.

        Supported formats:
            - "click [2]" -> {"click_element": {"index": 2}}
            - "input_text [3] hello world" -> {"input_text": {"index": 3, "text": "hello world"}}
            - "type_text(2, \"hello\")" -> {"input_text": {"index": 2, "text": "hello"}}
            - "type: hello, 2" -> {"input_text": {"index": 2, "text": "hello"}}
            - "type hello into input[name=foo]" -> {"input_text": {"css_selector": "input[name=foo]", "text": "hello"}}
            - "type_text #selector text" -> {"input_text": {"css_selector": "#selector", "text": "text"}}
            - "scroll down" -> {"scroll": {"down": True}}
            - "done task completed" -> {"done": {"text": "task completed"}}
            - Multiple actions: "type_text #id text click [7]" -> list of action dicts

        Args:
            action_str: The action string from inside <action>...</action> tags

        Returns:
            Action data dict, list of action dicts, or None if parsing fails
        """
        import re

        action_str = action_str.strip()

        # Just a number or [number] - interpret as click
        bare_index_match = re.match(r"^\[?(\d+)\]?$", action_str.strip())
        if bare_index_match:
            index = int(bare_index_match.group(1))
            return {"click_element": {"index": index}}

        # [index]Type "text" format - index prefix with action
        # e.g., [2]Type "hello" -> type "hello" into element 2
        index_type_match = re.match(r'^\[(\d+)\]\s*(?:type|input)\s*["\']([^"\']+)["\']$', action_str, re.IGNORECASE)
        if index_type_match:
            index = int(index_type_match.group(1))
            text = self._sanitize_text_value(index_type_match.group(2))
            return {"input_text": {"index": index, "text": text}}

        # [index]Click format - index prefix with click action (optional description)
        # e.g., [6]Click -> click element 6, [2]Click Log in -> click element 2
        index_click_match = re.match(r'^\[(\d+)\]\s*click(?:\s+.*)?$', action_str, re.IGNORECASE)
        if index_click_match:
            index = int(index_click_match.group(1))
            return {"click_element": {"index": index}}

        # click(index) - function call syntax for click
        click_func_match = re.match(r"click\s*\(\s*(\d+)\s*\)", action_str, re.IGNORECASE)
        if click_func_match:
            index = int(click_func_match.group(1))
            return {"click_element": {"index": index}}

        # click [index] or click index (with or without brackets)
        click_match = re.match(r"click\s*\[?(\d+)\]?", action_str, re.IGNORECASE)
        if click_match:
            index = int(click_match.group(1))
            return {"click_element": {"index": index}}

        # Click button/element with text "X" - natural language click
        # e.g., "Click button with text 'Next'" or "Click element with text 'Submit'"
        # Also handles without quotes: "Click button with text Next"
        click_text_match = re.match(
            r'click\s+(?:button|element|link|on)?\s*(?:with\s+)?text\s*[=:]?\s*["\']?([^"\']+?)["\']?\s*$',
            action_str, re.IGNORECASE
        )
        if click_text_match:
            button_text = click_text_match.group(1).strip()
            # Return a click with text selector - will need to be resolved by looking at DOM
            return {"click_element": {"text": button_text}}

        # Click button "X" - simpler format without "with text"
        # e.g., 'Click button "Next"' or 'Click button "Submit"'
        click_button_match = re.match(
            r'click\s+(?:button|element|link)\s*["\']([^"\']+)["\']',
            action_str, re.IGNORECASE
        )
        if click_button_match:
            button_text = click_button_match.group(1).strip()
            return {"click_element": {"text": button_text}}

        # type_text_in_element[index, text] - square bracket format with comma
        type_in_element_match = re.match(
            r'(?:type_text_in_element|type_in_element|input_in_element)\s*\[\s*(\d+)\s*,\s*([^\]]+)\]',
            action_str, re.IGNORECASE
        )
        if type_in_element_match:
            index = int(type_in_element_match.group(1))
            text = self._sanitize_text_value(type_in_element_match.group(2).strip().strip('"\''))
            return {"input_text": {"index": index, "text": text}}

        # Nested XML format: <type>set_element_value</type> <element_selector>...</element_selector> <value>...</value>
        nested_xml_type = re.search(r'<type>\s*(.*?)\s*</type>', action_str, re.IGNORECASE | re.DOTALL)
        if nested_xml_type:
            action_type = nested_xml_type.group(1).strip().lower()
            if action_type in ('set_element_value', 'type', 'input', 'input_text'):
                # Extract selector and value
                selector_match = re.search(r'<(?:element_selector|selector)>\s*(.*?)\s*</(?:element_selector|selector)>', action_str, re.IGNORECASE | re.DOTALL)
                value_match = re.search(r'<value>\s*(.*?)\s*</value>', action_str, re.IGNORECASE | re.DOTALL)
                if value_match:
                    text = self._sanitize_text_value(value_match.group(1).strip())
                    if selector_match:
                        selector = selector_match.group(1).strip()
                        # Check if selector looks like an index
                        index_check = re.match(r'^\[?(\d+)\]?$', selector)
                        if index_check:
                            return {"input_text": {"index": int(index_check.group(1)), "text": text}}
                        else:
                            return {"input_text": {"css_selector": selector, "text": text}}
            elif action_type == 'click':
                selector_match = re.search(r'<(?:element_selector|selector)>\s*(.*?)\s*</(?:element_selector|selector)>', action_str, re.IGNORECASE | re.DOTALL)
                if selector_match:
                    selector = selector_match.group(1).strip()
                    index_check = re.match(r'^\[?(\d+)\]?$', selector)
                    if index_check:
                        return {"click_element": {"index": int(index_check.group(1))}}
                    else:
                        return {"click_element": {"css_selector": selector}}

        # Key-value format: type: set_element_value element_selector: ... value: ...
        kv_type_match = re.search(r'type:\s*set_element_value', action_str, re.IGNORECASE)
        if kv_type_match:
            selector_match = re.search(r'element_selector:\s*(\S+)', action_str, re.IGNORECASE)
            value_match = re.search(r'value:\s*(.+?)(?:\s*$|\s+(?:element_selector|type):)', action_str, re.IGNORECASE)
            if value_match:
                text = self._sanitize_text_value(value_match.group(1).strip())
                if selector_match:
                    selector = selector_match.group(1).strip()
                    return {"input_text": {"css_selector": selector, "text": text}}

        # user_input/selector format: user_input='text', selector='...'
        user_input_match = re.search(r"user_input\s*=\s*['\"]([^'\"]*)['\"]", action_str, re.IGNORECASE)
        if user_input_match:
            text = self._sanitize_text_value(user_input_match.group(1))
            selector_match = re.search(r"selector\s*=\s*['\"]([^'\"]*)['\"]", action_str, re.IGNORECASE)
            if selector_match:
                selector = selector_match.group(1).strip()
                index_check = re.match(r'^\[?(\d+)\]?$', selector)
                if index_check:
                    return {"input_text": {"index": int(index_check.group(1)), "text": text}}
                else:
                    return {"input_text": {"css_selector": selector, "text": text}}

        # input_text [index] text OR type [index] text OR input index text (with or without brackets)
        input_match = re.match(r"(?:input_text|type|input)\s*\[?(\d+)\]?\s*(.*)", action_str, re.IGNORECASE | re.DOTALL)
        if input_match:
            index = int(input_match.group(1))
            text = self._sanitize_text_value(input_match.group(2).strip().strip('"\''))
            return {"input_text": {"index": index, "text": text}}

        # Function call syntax: type_text(index, "text") or type("index", "text")
        func_call_match = re.match(
            r'(?:type_text|type|input_text)\s*\(\s*["\']?(\d+|\[\d+\])["\']?\s*,\s*["\']([^"\']*)["\']',
            action_str, re.IGNORECASE
        )
        if func_call_match:
            index_str = func_call_match.group(1)
            # Handle [2] or just 2
            index = int(re.search(r'\d+', index_str).group())
            text = self._sanitize_text_value(func_call_match.group(2))
            return {"input_text": {"index": index, "text": text}}

        # Colon format: type: text, index OR type: text, selector
        colon_match = re.match(
            r'(?:type_text|type|input_text)\s*:\s*([^,]+),\s*(.+)',
            action_str, re.IGNORECASE
        )
        if colon_match:
            text = self._sanitize_text_value(colon_match.group(1).strip().strip('"\''))
            target = colon_match.group(2).strip().strip('"\'')
            # Check if target is an index
            if target.isdigit():
                return {"input_text": {"index": int(target), "text": text}}
            elif re.match(r'^\[\d+\]$', target):
                index = int(re.search(r'\d+', target).group())
                return {"input_text": {"index": index, "text": text}}
            else:
                # CSS selector or description - use as css_selector
                return {"input_text": {"css_selector": target, "text": text}}

        # "into" format: type text into selector (stop at newline to handle multi-line)
        into_match = re.match(
            r'(?:type_text|type|input_text)\s+(.+?)\s+into\s+([^\n]+)',
            action_str, re.IGNORECASE
        )
        if into_match:
            text = self._sanitize_text_value(into_match.group(1).strip().strip('"\''))
            selector = into_match.group(2).strip().strip('"\'')
            # Check if selector is an index
            if re.match(r'^\[\d+\]$', selector):
                index = int(re.search(r'\d+', selector).group())
                return {"input_text": {"index": index, "text": text}}
            else:
                # Check if selector is natural language with id reference
                # e.g., "input with id 'identifierId'" or "input with id \"identifierId\""
                id_match = re.search(r'(?:with\s+)?id\s*[=:"\']?\s*["\']?([^"\'>\s]+)["\']?', selector, re.IGNORECASE)
                if id_match:
                    element_id = id_match.group(1).strip('"\'')
                    return {"input_text": {"css_selector": f"#{element_id}", "text": text}}
                # Check for name attribute reference
                name_match = re.search(r'(?:with\s+)?name\s*[=:"\']?\s*["\']?([^"\'>\s]+)["\']?', selector, re.IGNORECASE)
                if name_match:
                    element_name = name_match.group(1).strip('"\'')
                    return {"input_text": {"css_selector": f"[name='{element_name}']", "text": text}}
                return {"input_text": {"css_selector": selector, "text": text}}

        # type_text #selector text OR type_text selector text (CSS selector with space-separated text)
        css_type_match = re.match(
            r'(?:type_text|input_text)\s+(#[\w-]+|\.[\w-]+|\[[\w\-=\'"]+\])\s+(.*)',
            action_str, re.IGNORECASE | re.DOTALL
        )
        if css_type_match:
            selector = css_type_match.group(1).strip()
            text = self._sanitize_text_value(css_type_match.group(2).strip().strip('"\''))
            return {"input_text": {"css_selector": selector, "text": text}}

        # Check for multiple actions in one string (e.g., "type_text #id text click [7]")
        # Look for action keywords and try to split
        multi_action_pattern = r'\b(click|type_text|type|input_text|scroll|done|go_back|navigate)\b'
        action_starts = [(m.start(), m.group(1)) for m in re.finditer(multi_action_pattern, action_str, re.IGNORECASE)]
        if len(action_starts) > 1:
            actions = []
            for i, (start, action_type) in enumerate(action_starts):
                end = action_starts[i + 1][0] if i + 1 < len(action_starts) else len(action_str)
                sub_action_str = action_str[start:end].strip()
                sub_action = self._parse_action_content(sub_action_str)
                if sub_action and not isinstance(sub_action, list):
                    actions.append(sub_action)
            if actions:
                return actions

        # scroll down/up
        scroll_match = re.match(r"scroll\s*(down|up)(?:\s+(\d+))?", action_str, re.IGNORECASE)
        if scroll_match:
            direction = scroll_match.group(1).lower()
            pages = int(scroll_match.group(2)) if scroll_match.group(2) else 1
            return {"scroll": {"down": direction == "down", "pages": pages}}

        # done [message]
        done_match = re.match(r"done\s*(.*)", action_str, re.IGNORECASE | re.DOTALL)
        if done_match:
            message = done_match.group(1).strip().strip('"\'') or "Task completed"
            return {"done": {"text": message}}

        # go_back
        if re.match(r"go_back", action_str, re.IGNORECASE):
            return {"go_back": {}}

        # navigate url - only match actual URLs (http/https or common domains)
        navigate_match = re.match(r"(?:navigate|goto|go_to|go\s+to)\s+[\"'`]?(https?://[^\s\"'`]+|www\.[^\s\"'`]+|\S+\.\S+/[^\s\"'`]*)[\"'`]?", action_str, re.IGNORECASE)
        if navigate_match:
            url = navigate_match.group(1).strip().strip('"\'`')
            # Validate it looks like a URL
            if '.' in url or url.startswith('http'):
                return {"navigate": {"url": url}}

        # send_keys keys
        keys_match = re.match(r"(?:send_keys|press)\s+(.*)", action_str, re.IGNORECASE)
        if keys_match:
            keys = keys_match.group(1).strip()
            return {"send_keys": {"keys": keys}}

        # wait [seconds]
        wait_match = re.match(r"wait\s*(\d+)?", action_str, re.IGNORECASE)
        if wait_match:
            seconds = int(wait_match.group(1)) if wait_match.group(1) else 3
            return {"wait": {"seconds": seconds}}

        # search query
        search_match = re.match(r"search\s+(.*)", action_str, re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip().strip('"\'')
            return {"search": {"query": query, "engine": "duckduckgo"}}

        # switch_tab / switch [tab_id]
        switch_match = re.match(r"(?:switch_tab|switch)\s+(.*)", action_str, re.IGNORECASE)
        if switch_match:
            tab_id = switch_match.group(1).strip().strip('"\'')
            return {"switch_tab": {"tab_id": tab_id}}

        # close_tab / close [tab_id]
        close_match = re.match(r"(?:close_tab|close)\s+(.*)", action_str, re.IGNORECASE)
        if close_match:
            tab_id = close_match.group(1).strip().strip('"\'')
            return {"close_tab": {"tab_id": tab_id}}

        # extract query
        extract_match = re.match(r"extract\s+(.*)", action_str, re.IGNORECASE | re.DOTALL)
        if extract_match:
            query = extract_match.group(1).strip().strip('"\'')
            return {"extract": {"query": query}}

        # find_text text
        find_text_match = re.match(r"find_text\s+(.*)", action_str, re.IGNORECASE)
        if find_text_match:
            text = find_text_match.group(1).strip().strip('"\'')
            return {"find_text": {"text": text}}

        # screenshot
        if re.match(r"screenshot", action_str, re.IGNORECASE):
            return {"screenshot": {}}

        # upload_file [index] path
        upload_match = re.match(r"upload_file\s*\[(\d+)\]\s+(.*)", action_str, re.IGNORECASE)
        if upload_match:
            index = int(upload_match.group(1))
            path = upload_match.group(2).strip().strip('"\'')
            return {"upload_file": {"index": index, "path": path}}

        # input [index] text (alias for input_text, used by browser-use)
        input_match = re.match(r"input\s*\[(\d+)\]\s*(.*)", action_str, re.IGNORECASE | re.DOTALL)
        if input_match:
            index = int(input_match.group(1))
            text = self._sanitize_text_value(input_match.group(2).strip().strip('"\''))
            return {"input_text": {"index": index, "text": text}}

        return None

    async def _convert_action_model(
        self, action_type: str, action_params: BaseModel
    ) -> Optional[AgentAction]:
        """Convert a validated Pydantic action model to AgentAction."""
        try:
            # Convert Pydantic model to dict for _map_action_to_stagehand
            params_dict = action_params.model_dump(exclude_unset=True)

            action_payload = await self._map_action_to_stagehand(action_type, params_dict)
            if not action_payload:
                return None

            # Validate and create action
            validated_action = TypeAdapter(AgentActionType).validate_python(
                action_payload
            )

            return AgentAction(
                action_type=action_payload.get("type", action_type),
                action=validated_action,
                reasoning=None,
            )

        except Exception as e:
            self.logger.error(f"Failed to convert action model: {e}")
            return None

    async def _convert_action(self, action_data: dict[str, Any]) -> Optional[AgentAction]:
        """Convert a browser-use action dict to AgentAction (fallback for unvalidated dicts)."""
        try:
            # Determine action type
            action_type = None
            action_params = None

            for key, value in action_data.items():
                if key in [
                    "click_element",
                    "input_text",
                    "scroll",
                    "navigate",
                    "go_back",
                    "send_keys",
                    "search",
                    "execute_script",
                    "wait",
                ]:
                    action_type = key
                    action_params = value if isinstance(value, dict) else {}
                    break

            if not action_type:
                return None

            action_payload = await self._map_action_to_stagehand(action_type, action_params)
            if not action_payload:
                return None

            # Validate and create action
            validated_action = TypeAdapter(AgentActionType).validate_python(
                action_payload
            )

            return AgentAction(
                action_type=action_payload.get("type", action_type),
                action=validated_action,
                reasoning=None,
            )

        except Exception as e:
            self.logger.error(f"Failed to convert action: {e}")
            return None

    async def _map_action_to_stagehand(
        self, action_type: str, params: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Map browser-use action to stagehand action format."""

        if action_type == "click_element":
            # Get coordinates from index, CSS selector, text, or direct coordinates
            index = params.get("index")
            css_selector = params.get("css_selector")
            text = params.get("text")  # Natural language text selector
            coord_x = params.get("coordinate_x")
            coord_y = params.get("coordinate_y")

            if index and self._current_dom_state:
                coords = self._current_dom_state.get_coordinates_for_index(index)
                if coords:
                    return {
                        "type": "click",
                        "x": coords[0],
                        "y": coords[1],
                        "button": "left",
                    }
            elif css_selector and self.handler and self.handler.page:
                # Use Playwright to find element by CSS selector
                try:
                    element = await self.handler.page.query_selector(css_selector)
                    if element:
                        box = await element.bounding_box()
                        if box:
                            # Click center of element
                            x = int(box["x"] + box["width"] / 2)
                            y = int(box["y"] + box["height"] / 2)
                            return {
                                "type": "click",
                                "x": x,
                                "y": y,
                                "button": "left",
                            }
                except Exception:
                    pass
            elif text and self.handler and self.handler.page:
                # Use Playwright's text selector to find element by visible text
                try:
                    # Try multiple selector strategies for text matching
                    selectors_to_try = [
                        f"text={text}",  # Playwright text selector
                        f"button:has-text('{text}')",  # Button with text
                        f"a:has-text('{text}')",  # Link with text
                        f"[type='submit']:has-text('{text}')",  # Submit button
                        f"input[value='{text}']",  # Input with value
                    ]
                    for selector in selectors_to_try:
                        try:
                            element = await self.handler.page.query_selector(selector)
                            if element:
                                box = await element.bounding_box()
                                if box:
                                    x = int(box["x"] + box["width"] / 2)
                                    y = int(box["y"] + box["height"] / 2)
                                    return {
                                        "type": "click",
                                        "x": x,
                                        "y": y,
                                        "button": "left",
                                    }
                        except Exception:
                            continue
                except Exception:
                    pass
            elif coord_x is not None and coord_y is not None:
                return {
                    "type": "click",
                    "x": int(coord_x),
                    "y": int(coord_y),
                    "button": "left",
                }

            return None

        elif action_type == "input_text":
            index = params.get("index")
            css_selector = params.get("css_selector")
            # Sanitize text as a final safeguard against XML artifacts
            text = self._sanitize_text_value(params.get("text", ""))

            # Get coordinates for the input element
            x, y = None, None
            if index and self._current_dom_state:
                coords = self._current_dom_state.get_coordinates_for_index(index)
                if coords:
                    x, y = coords
            elif css_selector and self.handler and self.handler.page:
                # Use Playwright to find element by CSS selector
                try:
                    element = await self.handler.page.query_selector(css_selector)
                    if element:
                        box = await element.bounding_box()
                        if box:
                            x = int(box["x"] + box["width"] / 2)
                            y = int(box["y"] + box["height"] / 2)
                except Exception:
                    pass

            return {
                "type": "type",
                "text": text,
                "x": x,
                "y": y,
            }

        elif action_type == "scroll":
            # Support both old format (down, pages) and new format (direction, amount)
            direction = params.get("direction", "down" if params.get("down", True) else "up")
            amount = params.get("amount", "page")
            pages = params.get("pages", 1)

            # Calculate scroll amount
            if amount == "page":
                scroll_pixels = self.viewport.get("height", 711) * pages
            else:
                scroll_pixels = int(pages * 800)

            # Negative for scrolling up
            if direction == "up":
                scroll_pixels = -abs(scroll_pixels)
            else:
                scroll_pixels = abs(scroll_pixels)

            # Scroll in center of viewport
            center_x = self.viewport.get("width", 1288) // 2
            center_y = self.viewport.get("height", 711) // 2

            return {
                "type": "scroll",
                "x": center_x,
                "y": center_y,
                "scroll_x": 0,
                "scroll_y": scroll_pixels,
            }

        elif action_type == "navigate":
            url = params.get("url", "")
            url = self._sanitize_text_value(url)
            return {
                "type": "function",
                "name": "goto",
                "arguments": FunctionArguments(url=url),
            }

        elif action_type == "go_back":
            return {
                "type": "function",
                "name": "navigate_back",
                "arguments": FunctionArguments(url=""),
            }

        elif action_type == "send_keys":
            keys_str = params.get("keys", "")
            keys = [self.key_to_playwright(k.strip()) for k in keys_str.split("+")]
            return {
                "type": "keypress",
                "keys": keys,
            }

        elif action_type == "search":
            query = params.get("query", "")
            engine = params.get("engine", "google").lower()

            search_urls = {
                "google": f"https://www.google.com/search?q={query}",
                "duckduckgo": f"https://duckduckgo.com/?q={query}",
                "bing": f"https://www.bing.com/search?q={query}",
            }
            url = search_urls.get(engine, search_urls["google"])

            return {
                "type": "function",
                "name": "goto",
                "arguments": FunctionArguments(url=url),
            }

        elif action_type == "execute_script":
            script = params.get("script", "")
            return {
                "type": "function",
                "name": "evaluate",
                "arguments": FunctionArguments(expression=script),
            }

        elif action_type == "wait":
            seconds = params.get("seconds", 2)
            return {
                "type": "wait",
                "miliseconds": int(seconds * 1000),
            }

        return None

    def _format_action_feedback(
        self,
        action: AgentAction,
        action_result: dict[str, Any],
        new_screenshot_base64: str,
    ) -> list[dict[str, Any]]:
        """Format feedback after action execution.

        Note: action_result is a dict with 'success' and optional 'error' keys,
        returned by CUAHandler.perform_action().
        """
        feedback_content = []

        # Add action result - handle dict access
        success = action_result.get("success", False) if isinstance(action_result, dict) else getattr(action_result, "success", False)
        error = action_result.get("error") if isinstance(action_result, dict) else getattr(action_result, "error", None)

        if success:
            feedback_content.append({
                "type": "text",
                "text": "Action executed successfully.",
            })
        else:
            feedback_content.append({
                "type": "text",
                "text": f"Action failed: {error or 'Unknown error'}",
            })

        # Add new DOM state
        if self._current_dom_state:
            dom_text = self._current_dom_state.serialize(DEFAULT_INCLUDE_ATTRIBUTES)
            feedback_content.append({
                "type": "text",
                "text": f"<browser_state>\n{dom_text}\n</browser_state>",
            })

            feedback_content.append({
                "type": "text",
                "text": f"Current URL: {self._current_dom_state.url}",
            })

        # Add screenshot (resized/compressed to reduce payload size)
        if new_screenshot_base64:
            resized_screenshot = self._resize_screenshot(new_screenshot_base64)
            feedback_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{resized_screenshot}",
                    "media_type": "image/jpeg",
                    "detail": "auto",
                },
            })

        return [{"role": "user", "content": feedback_content}]

    async def _make_api_call(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Make API call to browser-use with retry logic.

        Passes output_format schema so the API returns structured JSON
        that we can validate with Pydantic.
        """
        # Note: output_format is intentionally NOT included because it causes
        # significant API instability (~60% failure rate vs 0% without it).
        # We rely on string parsing of the response instead.
        payload = {
            "model": self.model,
            "messages": messages,
            "request_type": "browser_agent",
            "fast": False,  # Required by browser-use API
            "anonymized_telemetry": True,  # Required by browser-use API
            "session_id": self.session_id,  # Sticky routing for API stability
        }

        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    response.raise_for_status()
                    return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code

                if status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries - 1:
                    delay = min(1.0 * (2**attempt), 60.0)
                    jitter = random.uniform(0, delay * 0.1)
                    self.logger.info(
                        f"Got {status_code}, retrying in {delay + jitter:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay + jitter)
                    continue

                # Non-retryable error
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("detail", str(e))
                except Exception:
                    error_detail = str(e)

                raise ValueError(f"API error ({status_code}): {error_detail}")

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = min(1.0 * (2**attempt), 60.0)
                    self.logger.info(f"Network error, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})...")
                    await asyncio.sleep(delay)
                    continue

                raise ValueError(f"Network error after {self.max_retries} attempts: {e}")

        raise ValueError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def _resize_screenshot(self, screenshot_b64: str) -> str:
        """Resize screenshot to reduce payload size if needed.

        Only downsizes - never upscales. Re-encodes as JPEG with quality 85
        for significant size reduction (typically 5-10x smaller than PNG).
        """
        img = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))
        original_size = img.size

        # Calculate target size maintaining aspect ratio, only if larger
        target_w, target_h = self.LLM_SCREENSHOT_SIZE
        orig_w, orig_h = original_size

        # Only resize if larger than target
        if orig_w > target_w or orig_h > target_h:
            # Scale to fit within target dimensions while maintaining aspect ratio
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to RGB (JPEG doesn't support alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Encode as JPEG for much smaller size
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
        resized_b64 = base64.b64encode(buffer.getvalue()).decode()

        return resized_b64

    def format_screenshot(self, screenshot_base64: str) -> dict[str, Any]:
        """Format screenshot for browser-use API (with compression)."""
        resized = self._resize_screenshot(screenshot_base64)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{resized}",
                "media_type": "image/jpeg",
                "detail": "auto",
            },
        }

    def key_to_playwright(self, key: str) -> str:
        """Convert browser-use key name to Playwright key."""
        return self.KEY_MAPPING.get(key.lower(), key)
