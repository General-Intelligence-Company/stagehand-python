"""BrowserUse CUA Client for Stagehand.

This client uses the browser-use cloud API (ChatBrowserUse) as the model provider
for computer use agent tasks, with DOM extraction for optimal model accuracy.
"""

import asyncio
import os
import random
from typing import Any, Optional

import httpx
from pydantic import TypeAdapter

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

            # Make API call
            start_time = asyncio.get_event_loop().time()
            try:
                response = await self._make_api_call(messages)
                end_time = asyncio.get_event_loop().time()
                total_inference_time_ms += int((end_time - start_time) * 1000)

                # Extract usage
                usage = response.get("usage", {})
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)

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
            ) = self._process_provider_response(response)

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
                for agent_action in agent_actions:
                    actions_taken.append(agent_action)

                    # Execute the action
                    action_result: ActionExecutionResult = (
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
                # No actions returned
                self.logger.info(
                    "Model did not return any actions. Ending task.",
                    category=StagehandFunctionName.AGENT,
                )
                break

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

        Returns:
            - List of AgentActions to execute
            - Reasoning text
            - Whether task is done
            - Done message (if task is done)
        """
        completion = response.get("completion", {})

        # Extract reasoning
        thinking = completion.get("thinking")
        next_goal = completion.get("next_goal")
        memory = completion.get("memory")

        reasoning_parts = []
        if thinking:
            reasoning_parts.append(f"Thinking: {thinking}")
        if next_goal:
            reasoning_parts.append(f"Goal: {next_goal}")
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else None

        # Extract actions
        actions_data = completion.get("action", [])
        if not isinstance(actions_data, list):
            actions_data = [actions_data] if actions_data else []

        agent_actions = []
        is_done = False
        done_message = None

        for action_data in actions_data:
            if not isinstance(action_data, dict):
                continue

            # Check for done action
            if "done" in action_data:
                is_done = True
                done_info = action_data["done"]
                if isinstance(done_info, dict):
                    done_message = done_info.get("text", "Task completed")
                else:
                    done_message = str(done_info)
                continue

            # Convert to AgentAction
            agent_action = self._convert_action(action_data)
            if agent_action:
                agent_actions.append(agent_action)

        return agent_actions, reasoning, is_done, done_message

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
        action_result: ActionExecutionResult,
        new_screenshot_base64: str,
    ) -> list[dict[str, Any]]:
        """Format feedback after action execution."""
        feedback_content = []

        # Add action result
        if action_result.get("success", False):
            feedback_content.append({
                "type": "text",
                "text": "Action executed successfully.",
            })
        else:
            error = action_result.get("error", "Unknown error")
            feedback_content.append({
                "type": "text",
                "text": f"Action failed: {error}",
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
        """Make API call to browser-use with retry logic."""
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
