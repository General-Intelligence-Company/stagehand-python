"""BrowserUse CUA Client for Stagehand.

This client uses the browser-use cloud API (ChatBrowserUse) as the model provider
for computer use agent tasks, with DOM extraction for optimal model accuracy.
"""

import asyncio
import os
import random
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, TypeAdapter

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


# Output format schema for browser-use API (matches browser-use's AgentOutput)
class BrowserUseActionModel(BaseModel):
    """Dynamic action model - one of these will be set."""
    click_element: Optional[dict[str, Any]] = Field(default=None, description="Click an element by index")
    input_text: Optional[dict[str, Any]] = Field(default=None, description="Type text into an element")
    scroll: Optional[dict[str, Any]] = Field(default=None, description="Scroll the page")
    navigate: Optional[dict[str, Any]] = Field(default=None, description="Navigate to a URL")
    go_back: Optional[dict[str, Any]] = Field(default=None, description="Go back in browser history")
    send_keys: Optional[dict[str, Any]] = Field(default=None, description="Send keyboard keys")
    done: Optional[dict[str, Any]] = Field(default=None, description="Mark task as done")


class BrowserUseAgentOutput(BaseModel):
    """Output format for browser-use API responses."""
    thinking: Optional[str] = Field(default=None, description="Model's internal reasoning")
    evaluation_previous_goal: Optional[str] = Field(default=None, description="Evaluation of previous goal")
    memory: Optional[str] = Field(default=None, description="What to remember")
    next_goal: Optional[str] = Field(default=None, description="Next goal to achieve")
    action: list[dict[str, Any]] = Field(
        ...,
        description="List of actions to execute",
        min_length=1
    )

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

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

        # API configuration
        self.api_key = (
            config.options.get("apiKey") if config and config.options else None
        ) or os.getenv("BROWSER_USE_API_KEY")

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
        self.max_retries = kwargs.get("max_retries", 5)

        self.logger.info(
            f"BrowserUseCUAClient initialized for model: {self.model}",
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
        self.logger.info(
            f"[DEBUG] run_task called with instruction: '{instruction}'",
            category=StagehandFunctionName.AGENT,
        )

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

        self.logger.info(
            "[DEBUG] Handler available, injecting cursor...",
            category=StagehandFunctionName.AGENT,
        )

        # Inject cursor for visual feedback
        await self.handler.inject_cursor()

        self.logger.info(
            "[DEBUG] Getting initial screenshot...",
            category=StagehandFunctionName.AGENT,
        )

        # Get initial state
        current_screenshot_b64 = await self.handler.get_screenshot_base64()

        self.logger.info(
            f"[DEBUG] Screenshot obtained, length: {len(current_screenshot_b64) if current_screenshot_b64 else 0}",
            category=StagehandFunctionName.AGENT,
        )

        self.logger.info(
            "[DEBUG] Getting DOM service...",
            category=StagehandFunctionName.AGENT,
        )

        dom_service = await self._get_dom_service()

        self.logger.info(
            "[DEBUG] Getting DOM state...",
            category=StagehandFunctionName.AGENT,
        )

        self._current_dom_state = await dom_service.get_dom_state()

        self.logger.info(
            f"[DEBUG] DOM state obtained: {self._current_dom_state is not None}",
            category=StagehandFunctionName.AGENT,
        )

        self.logger.info(
            "[DEBUG] Formatting initial messages...",
            category=StagehandFunctionName.AGENT,
        )

        # Format initial messages
        messages = self._format_initial_messages(
            instruction, current_screenshot_b64
        )

        self.logger.info(
            f"[DEBUG] Initial messages formatted, count: {len(messages)}",
            category=StagehandFunctionName.AGENT,
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

            # Make API call
            self.logger.info(
                "[DEBUG] Making API call...",
                category=StagehandFunctionName.AGENT,
            )

            start_time = asyncio.get_event_loop().time()
            try:
                response = await self._make_api_call(messages)
                end_time = asyncio.get_event_loop().time()
                total_inference_time_ms += int((end_time - start_time) * 1000)

                self.logger.info(
                    f"[DEBUG] API response received. Type: {type(response).__name__}",
                    category=StagehandFunctionName.AGENT,
                )
                self.logger.info(
                    f"[DEBUG] API response content: {str(response)[:500]}...",
                    category=StagehandFunctionName.AGENT,
                )

                # Extract usage - with safety check
                if isinstance(response, dict):
                    usage = response.get("usage", {})
                    total_input_tokens += usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                    total_output_tokens += usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
                else:
                    self.logger.error(
                        f"[DEBUG] Response is not a dict! Type: {type(response).__name__}, Value: {response}",
                        category=StagehandFunctionName.AGENT,
                    )

            except Exception as e:
                self.logger.error(
                    f"BrowserUse API call failed: {e}",
                    category=StagehandFunctionName.AGENT,
                )
                import traceback
                self.logger.error(
                    f"[DEBUG] Full traceback: {traceback.format_exc()}",
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
            self.logger.info(
                "[DEBUG] Processing provider response...",
                category=StagehandFunctionName.AGENT,
            )

            (
                agent_actions,
                reasoning,
                is_done,
                done_message,
            ) = self._process_provider_response(response)

            self.logger.info(
                f"[DEBUG] Processed response - actions: {len(agent_actions)}, reasoning: {reasoning is not None}, is_done: {is_done}",
                category=StagehandFunctionName.AGENT,
            )

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
                self.logger.info(
                    f"[DEBUG] Executing {len(agent_actions)} actions...",
                    category=StagehandFunctionName.AGENT,
                )

                for idx, agent_action in enumerate(agent_actions):
                    self.logger.info(
                        f"[DEBUG] Executing action {idx + 1}/{len(agent_actions)}: {agent_action}",
                        category=StagehandFunctionName.AGENT,
                    )

                    actions_taken.append(agent_action)

                    # Execute the action
                    self.logger.info(
                        "[DEBUG] Calling handler.perform_action...",
                        category=StagehandFunctionName.AGENT,
                    )

                    # Note: perform_action returns a dict, not ActionExecutionResult
                    action_result: dict[str, Any] = (
                        await self.handler.perform_action(agent_action)
                    )

                    self.logger.info(
                        f"[DEBUG] Action result type: {type(action_result).__name__}",
                        category=StagehandFunctionName.AGENT,
                    )
                    self.logger.info(
                        f"[DEBUG] Action result: {action_result}",
                        category=StagehandFunctionName.AGENT,
                    )

                    # Get new state after action
                    self.logger.info(
                        "[DEBUG] Getting new screenshot after action...",
                        category=StagehandFunctionName.AGENT,
                    )

                    current_screenshot_b64 = await self.handler.get_screenshot_base64()
                    self._current_dom_state = await dom_service.get_dom_state()

                    # Format feedback
                    self.logger.info(
                        "[DEBUG] Formatting action feedback...",
                        category=StagehandFunctionName.AGENT,
                    )

                    feedback = self._format_action_feedback(
                        action=agent_action,
                        action_result=action_result,
                        new_screenshot_base64=current_screenshot_b64,
                    )

                    self.logger.info(
                        f"[DEBUG] Feedback formatted, extending messages with {len(feedback)} items",
                        category=StagehandFunctionName.AGENT,
                    )

                    messages.extend(feedback)

            else:
                # No actions returned
                self.logger.info(
                    "Model did not return any actions. Ending task.",
                    category=StagehandFunctionName.AGENT,
                )
                break

        self.logger.info(
            f"[DEBUG] run_task completing. completed={task_completed}, actions={len(actions_taken)}",
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

        # Add screenshot
        if screenshot_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_base64}",
                },
            })

        messages.append({"role": "user", "content": user_content})

        return messages

    def _process_provider_response(
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
        self.logger.info(
            f"[DEBUG] _process_provider_response called. response type: {type(response).__name__}",
            category=StagehandFunctionName.AGENT,
        )

        if not isinstance(response, dict):
            self.logger.error(
                f"[DEBUG] Response is not a dict! Type: {type(response).__name__}, Value: {str(response)[:200]}",
                category=StagehandFunctionName.AGENT,
            )
            return [], None, False, None

        completion = response.get("completion", {})
        self.logger.info(
            f"[DEBUG] completion type: {type(completion).__name__}, value: {str(completion)[:500]}",
            category=StagehandFunctionName.AGENT,
        )

        # Handle structured dict response (when output_format is provided)
        if isinstance(completion, dict):
            return self._process_structured_completion(completion)

        # Fallback: Handle string response with XML tags (when output_format is not used)
        if isinstance(completion, str):
            return self._process_string_completion(completion)

        self.logger.error(
            f"[DEBUG] completion is neither dict nor string! Type: {type(completion).__name__}",
            category=StagehandFunctionName.AGENT,
        )
        return [], None, False, None

    def _process_structured_completion(
        self, completion: dict[str, Any]
    ) -> tuple[list[AgentAction], Optional[str], bool, Optional[str]]:
        """Process structured completion dict from browser-use API."""
        # Extract reasoning from structured fields
        thinking = completion.get("thinking")
        memory = completion.get("memory")
        next_goal = completion.get("next_goal")
        eval_prev = completion.get("evaluation_previous_goal")

        reasoning_parts = []
        if thinking:
            reasoning_parts.append(f"Thinking: {thinking}")
        if next_goal:
            reasoning_parts.append(f"Goal: {next_goal}")
        if memory:
            reasoning_parts.append(f"Memory: {memory}")
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else None

        self.logger.info(
            f"[DEBUG] Structured completion - thinking: {thinking is not None}, memory: {memory is not None}, next_goal: {next_goal is not None}",
            category=StagehandFunctionName.AGENT,
        )

        # Extract actions
        action_list = completion.get("action", [])
        if not isinstance(action_list, list):
            action_list = [action_list] if action_list else []

        self.logger.info(
            f"[DEBUG] Structured completion has {len(action_list)} actions: {action_list}",
            category=StagehandFunctionName.AGENT,
        )

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
                self.logger.info(
                    f"[DEBUG] Done action detected: {done_message}",
                    category=StagehandFunctionName.AGENT,
                )
                continue

            # Convert to AgentAction
            agent_action = self._convert_action(action_data)
            if agent_action:
                agent_actions.append(agent_action)

        return agent_actions, reasoning, is_done, done_message

    def _process_string_completion(
        self, completion: str
    ) -> tuple[list[AgentAction], Optional[str], bool, Optional[str]]:
        """Process string completion with XML action tags (fallback mode)."""
        # Parse the completion string to extract reasoning and actions
        reasoning, action_dicts = self._parse_completion_string(completion)

        self.logger.info(
            f"[DEBUG] String completion parsed - reasoning: {reasoning[:100] if reasoning else None}...",
            category=StagehandFunctionName.AGENT,
        )
        self.logger.info(
            f"[DEBUG] String completion parsed - action dicts: {action_dicts}",
            category=StagehandFunctionName.AGENT,
        )

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
                self.logger.info(
                    f"[DEBUG] Done action detected: {done_message}",
                    category=StagehandFunctionName.AGENT,
                )
                continue

            # Convert to AgentAction
            agent_action = self._convert_action(action_data)
            if agent_action:
                agent_actions.append(agent_action)

        return agent_actions, reasoning, is_done, done_message

    def _parse_completion_string(self, completion: str) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Parse the completion string to extract reasoning and actions.

        Handles two action formats:
        1. Self-closing XML: <action type="click" selector="[2]"/>
        2. Content-based: <action>click [2]</action>

        Args:
            completion: String like 'Reasoning...\n\n<action type="click" selector="[2]"/>'

        Returns:
            - Reasoning text (everything before first <action> tag)
            - List of parsed action dicts
        """
        import re

        self.logger.info(
            f"[DEBUG] _parse_completion_string input: '{completion}'",
            category=StagehandFunctionName.AGENT,
        )

        actions = []

        # Pattern 1: Self-closing XML tags like <action type="click" selector="[2]"/>
        # This matches <action followed by attributes and ending with />
        self_closing_pattern = r'<action\s+([^>]*?)\s*/>'
        self_closing_matches = re.finditer(self_closing_pattern, completion, re.DOTALL)

        for match in self_closing_matches:
            attrs_str = match.group(1)
            self.logger.info(
                f"[DEBUG] Found self-closing action tag with attrs: '{attrs_str}'",
                category=StagehandFunctionName.AGENT,
            )
            action_dict = self._parse_action_attributes(attrs_str)
            if action_dict:
                actions.append(action_dict)

        # Pattern 2: Content-based tags like <action>click [2]</action>
        content_pattern = r'<action>(.*?)</action>'
        content_matches = re.findall(content_pattern, completion, re.DOTALL)

        for content in content_matches:
            self.logger.info(
                f"[DEBUG] Found content-based action tag: '{content}'",
                category=StagehandFunctionName.AGENT,
            )
            action_dict = self._parse_action_content(content.strip())
            if action_dict:
                actions.append(action_dict)

        # Extract reasoning (everything before the first <action tag)
        first_action_pos = completion.find("<action")
        if first_action_pos > 0:
            reasoning = completion[:first_action_pos].strip()
        elif first_action_pos == 0:
            reasoning = None
        else:
            # No action tags found
            reasoning = completion.strip() if completion.strip() else None

        self.logger.info(
            f"[DEBUG] Parsed {len(actions)} actions, reasoning: {reasoning[:100] if reasoning else None}...",
            category=StagehandFunctionName.AGENT,
        )

        return reasoning, actions

    def _parse_action_attributes(self, attrs_str: str) -> Optional[dict[str, Any]]:
        """Parse action attributes from a self-closing tag.

        Example: type="click" selector="[2]"

        Args:
            attrs_str: The attributes string from inside the tag

        Returns:
            Action data dict or None if parsing fails
        """
        import re

        self.logger.info(
            f"[DEBUG] Parsing action attributes: '{attrs_str}'",
            category=StagehandFunctionName.AGENT,
        )

        # Extract all attribute key="value" pairs
        attr_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
        attrs = dict(re.findall(attr_pattern, attrs_str))

        self.logger.info(
            f"[DEBUG] Extracted attributes: {attrs}",
            category=StagehandFunctionName.AGENT,
        )

        action_type = attrs.get("type", "").lower()

        if action_type == "click":
            # selector="[2]" -> extract index 2
            selector = attrs.get("selector", "")
            index_match = re.search(r'\[(\d+)\]', selector)
            if index_match:
                index = int(index_match.group(1))
                return {"click_element": {"index": index}}
            # Maybe selector is just a number
            if selector.isdigit():
                return {"click_element": {"index": int(selector)}}

        elif action_type in ("input_text", "type", "input"):
            selector = attrs.get("selector", "")
            text = attrs.get("text", attrs.get("value", ""))
            index_match = re.search(r'\[(\d+)\]', selector)
            if index_match:
                index = int(index_match.group(1))
                return {"input_text": {"index": index, "text": text}}

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
            return {"navigate": {"url": url}}

        elif action_type in ("go_back", "back"):
            return {"go_back": {}}

        elif action_type in ("send_keys", "press", "key"):
            keys = attrs.get("keys", attrs.get("key", ""))
            return {"send_keys": {"keys": keys}}

        self.logger.warning(
            f"[DEBUG] Unknown action type in attributes: '{action_type}' from '{attrs_str}'",
            category=StagehandFunctionName.AGENT,
        )
        return None

    def _parse_action_content(self, action_str: str) -> Optional[dict[str, Any]]:
        """Parse an action string from content-based tags.

        Supported formats:
            - "click [2]" -> {"click_element": {"index": 2}}
            - "input_text [3] hello world" -> {"input_text": {"index": 3, "text": "hello world"}}
            - "scroll down" -> {"scroll": {"down": True}}
            - "scroll up" -> {"scroll": {"down": False}}
            - "done task completed" -> {"done": {"text": "task completed"}}
            - "go_back" -> {"go_back": {}}
            - "navigate https://..." -> {"navigate": {"url": "https://..."}}

        Args:
            action_str: The action string from inside <action>...</action> tags

        Returns:
            Action data dict or None if parsing fails
        """
        import re

        action_str = action_str.strip()

        self.logger.info(
            f"[DEBUG] Parsing action content: '{action_str}'",
            category=StagehandFunctionName.AGENT,
        )

        # click [index]
        click_match = re.match(r"click\s*\[(\d+)\]", action_str, re.IGNORECASE)
        if click_match:
            index = int(click_match.group(1))
            return {"click_element": {"index": index}}

        # input_text [index] text OR type [index] text
        input_match = re.match(r"(?:input_text|type)\s*\[(\d+)\]\s*(.*)", action_str, re.IGNORECASE | re.DOTALL)
        if input_match:
            index = int(input_match.group(1))
            text = input_match.group(2).strip().strip('"\'')
            return {"input_text": {"index": index, "text": text}}

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

        # navigate url
        navigate_match = re.match(r"(?:navigate|goto|go_to)\s+(.*)", action_str, re.IGNORECASE)
        if navigate_match:
            url = navigate_match.group(1).strip().strip('"\'')
            return {"navigate": {"url": url}}

        # send_keys keys
        keys_match = re.match(r"(?:send_keys|press)\s+(.*)", action_str, re.IGNORECASE)
        if keys_match:
            keys = keys_match.group(1).strip()
            return {"send_keys": {"keys": keys}}

        self.logger.warning(
            f"[DEBUG] Could not parse action content: '{action_str}'",
            category=StagehandFunctionName.AGENT,
        )
        return None

    def _convert_action(self, action_data: dict[str, Any]) -> Optional[AgentAction]:
        """Convert a browser-use action to AgentAction."""
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
                ]:
                    action_type = key
                    action_params = value if isinstance(value, dict) else {}
                    break

            if not action_type:
                self.logger.warning(f"Unknown action format: {action_data}")
                return None

            action_payload = self._map_action_to_stagehand(action_type, action_params)
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

    def _map_action_to_stagehand(
        self, action_type: str, params: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Map browser-use action to stagehand action format."""

        if action_type == "click_element":
            # Get coordinates from index or direct coordinates
            index = params.get("index")
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
            elif coord_x is not None and coord_y is not None:
                return {
                    "type": "click",
                    "x": int(coord_x),
                    "y": int(coord_y),
                    "button": "left",
                }

            self.logger.warning(f"Could not resolve click coordinates for: {params}")
            return None

        elif action_type == "input_text":
            index = params.get("index")
            text = params.get("text", "")

            # Get coordinates for the input element
            x, y = None, None
            if index and self._current_dom_state:
                coords = self._current_dom_state.get_coordinates_for_index(index)
                if coords:
                    x, y = coords

            return {
                "type": "type",
                "text": text,
                "x": x,
                "y": y,
            }

        elif action_type == "scroll":
            down = params.get("down", True)
            pages = params.get("pages", 1)

            # Convert pages to pixels (approx 800px per page)
            scroll_amount = int(pages * 800)
            if not down:
                scroll_amount = -scroll_amount

            # Scroll in center of viewport
            center_x = self.viewport.get("width", 1288) // 2
            center_y = self.viewport.get("height", 711) // 2

            return {
                "type": "scroll",
                "x": center_x,
                "y": center_y,
                "scroll_x": 0,
                "scroll_y": scroll_amount,
            }

        elif action_type == "navigate":
            url = params.get("url", "")
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

        # Add screenshot
        if new_screenshot_base64:
            feedback_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{new_screenshot_base64}",
                },
            })

        return [{"role": "user", "content": feedback_content}]

    async def _make_api_call(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Make API call to browser-use with retry logic.

        Note: We don't pass output_format because the browser-use API expects
        a specific dynamically-generated schema from their tools registry.
        Without output_format, we get raw text with XML action tags which
        we parse in _process_string_completion.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "request_type": "browser_agent",
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
                    self.logger.warning(
                        f"Got {status_code}, retrying in {delay + jitter:.1f}s..."
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
                    self.logger.warning(f"Network error, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue

                raise ValueError(f"Network error after {self.max_retries} attempts: {e}")

        raise ValueError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def format_screenshot(self, screenshot_base64: str) -> dict[str, Any]:
        """Format screenshot for browser-use API."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{screenshot_base64}",
            },
        }

    def key_to_playwright(self, key: str) -> str:
        """Convert browser-use key name to Playwright key."""
        return self.KEY_MAPPING.get(key.lower(), key)
