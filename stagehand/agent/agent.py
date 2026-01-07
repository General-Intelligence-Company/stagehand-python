from typing import Optional, Union

from ..handlers.cua_handler import CUAHandler
from ..schemas import (
    AgentExecuteResult,
    AgentProvider,
)
from ..types.agent import (
    AgentConfig,
    AgentExecuteOptions,
    AgentResult,
    AgentUsage,
)
from .anthropic_cua import AnthropicCUAClient
from .browseruse_cua import BrowserUseCUAClient
from .client import AgentClient
from .google_cua import GoogleCUAClient
from .openai_cua import OpenAICUAClient

# Default viewport dimensions (matches TypeScript SDK)
DEFAULT_VIEWPORT_WIDTH = 1288
DEFAULT_VIEWPORT_HEIGHT = 711

MODEL_TO_CLIENT_CLASS_MAP: dict[str, type[AgentClient]] = {
    "computer-use-preview-2025-03-11": OpenAICUAClient,
    "claude-3-5-sonnet-latest": AnthropicCUAClient,
    "claude-3-7-sonnet-latest": AnthropicCUAClient,
    "claude-haiku-4-5-20251001": AnthropicCUAClient,
    "claude-sonnet-4-20250514": AnthropicCUAClient,
    "claude-sonnet-4-5-20250929": AnthropicCUAClient,
    "claude-opus-4-5-20251101": AnthropicCUAClient,
    "gemini-2.5-computer-use-preview-10-2025": GoogleCUAClient,
    # Browser-use models
    "bu-latest": BrowserUseCUAClient,
    "bu-1-0": BrowserUseCUAClient,
}
MODEL_TO_PROVIDER_MAP: dict[str, AgentProvider] = {
    "computer-use-preview-2025-03-11": AgentProvider.OPENAI,
    "claude-3-5-sonnet-20240620": AgentProvider.ANTHROPIC,
    "claude-3-7-sonnet-20250219": AgentProvider.ANTHROPIC,
    "claude-haiku-4-5-20251001": AgentProvider.ANTHROPIC,
    "claude-sonnet-4-20250514": AgentProvider.ANTHROPIC,
    "claude-sonnet-4-5-20250929": AgentProvider.ANTHROPIC,
    "claude-opus-4-5-20251101": AgentProvider.ANTHROPIC,
    "gemini-2.5-computer-use-preview-10-2025": AgentProvider.GOOGLE,
    # Browser-use models
    "bu-latest": AgentProvider.BROWSER_USE,
    "bu-1-0": AgentProvider.BROWSER_USE,
}

AGENT_METRIC_FUNCTION_NAME = "AGENT_EXECUTE_TASK"


class Agent:

    def __init__(self, stagehand_client, **kwargs):
        self.stagehand = stagehand_client
        self.config = AgentConfig(**kwargs) if kwargs else AgentConfig()
        self.logger = self.stagehand.logger

        # Determine provider from model
        if self.config.model in MODEL_TO_PROVIDER_MAP:
            self.provider = MODEL_TO_PROVIDER_MAP[self.config.model]
        else:
            self.provider = None
            self.logger.error(
                f"Could not infer provider for model: {self.config.model}"
            )

        # browser_use provider handles its own API calls, so it needs local client setup
        # even when use_api=True
        needs_local_client = not self.stagehand.use_api or self.provider == AgentProvider.BROWSER_USE

        if needs_local_client:
            if not hasattr(self.stagehand, "page") or not hasattr(
                self.stagehand.page, "_page"
            ):
                self.logger.error(
                    "Stagehand page object not available for CUAHandler initialization."
                )
                raise ValueError("Stagehand page not initialized. Cannot create Agent.")

            self.cua_handler = CUAHandler(
                stagehand=self.stagehand,
                page=self.stagehand.page._page,
                logger=self.logger,
            )

            # viewport_size can be None when connecting via CDP (e.g., Browserbase)
            # We'll get the real viewport from window.innerWidth/innerHeight in _init_viewport()
            self.viewport = self.stagehand.page._page.viewport_size
            self.client: AgentClient = self._get_client()

    def _get_client(self) -> AgentClient:
        ClientClass = MODEL_TO_CLIENT_CLASS_MAP.get(self.config.model)  # noqa: N806
        if not ClientClass:
            self.logger.error(
                f"Unsupported model or client not mapped: {self.config.model}"
            )
            raise ValueError(
                f"Unsupported model or client not mapped: {self.config.model}"
            )

        return ClientClass(
            model=self.config.model,
            instructions=(
                self.config.instructions
                if self.config.instructions
                else "Your browser is in full screen mode. There is no search bar, or navigation bar, or shortcut to control it. You can use the goto tool to navigate to different urls. Do not try to access a top navigation bar or other browser features."
            ),
            config=self.config,
            logger=self.logger,
            handler=self.cua_handler,
            viewport=self.viewport,
            experimental=self.stagehand.experimental,
        )

    async def _update_viewport_from_page(self) -> None:
        """
        Update the client's viewport dimensions by evaluating window.innerWidth/innerHeight
        in the browser. This is essential for Browserbase/CDP connections where Playwright's
        viewport_size may be None or incorrect.
        
        This matches the TypeScript SDK's updateClientViewport() behavior.
        """
        try:
            page = self.stagehand.page._page
            # Evaluate window dimensions directly in the page (like TypeScript does)
            dimensions = await page.evaluate(
                "({ width: window.innerWidth, height: window.innerHeight })"
            )
            
            if dimensions and dimensions.get("width") and dimensions.get("height"):
                width = dimensions["width"]
                height = dimensions["height"]
                self.viewport = {"width": width, "height": height}
                
                # Update the client's viewport dimensions using set_viewport if available
                # This is important for GoogleCUAClient's coordinate normalization
                if hasattr(self.client, "set_viewport"):
                    self.client.set_viewport(width, height)
                    self.logger.info(
                        f"Updated client viewport from window.innerWidth/Height: {width}x{height}",
                        category="agent",
                    )
                # Fallback for clients that use display_width/display_height directly
                elif hasattr(self.client, "display_width"):
                    self.client.display_width = width
                    self.client.display_height = height
                    self.logger.debug(
                        f"Updated viewport from page: {width}x{height}",
                        category="agent",
                    )
        except Exception as e:
            self.logger.debug(
                f"Could not update viewport from page: {e}",
                category="agent",
            )

    async def execute(
        self,
        options_or_instruction: Union[AgentExecuteOptions, str, dict, None] = None,
        **kwargs,
    ) -> AgentResult:
        options: Optional[AgentExecuteOptions] = None
        options_dict = {}

        if isinstance(options_or_instruction, AgentExecuteOptions):
            options_dict = options_or_instruction.model_dump()
        elif isinstance(options_or_instruction, dict):
            options_dict = options_or_instruction.copy()
        elif isinstance(options_or_instruction, str):
            options_dict["instruction"] = options_or_instruction

        options_dict.update(kwargs)

        try:
            options = AgentExecuteOptions(**options_dict)
        except Exception as e:
            self.logger.error(f"Invalid agent execute options: {e}")
            raise

        if not options.instruction:
            self.logger.error("No instruction provided for agent execution.")
            return AgentResult(
                message="No instruction provided.",
                completed=True,
                actions=[],
                usage={},
            )

        instruction = options.instruction

        # browser_use provider handles its own API calls directly to browser-use cloud,
        # so we should use the local client path even if use_api is True
        use_local_client = not self.stagehand.use_api or self.provider == AgentProvider.BROWSER_USE

        if use_local_client:
            self.logger.info(
                f"Agent starting execution for instruction: '{instruction}'",
                category="agent",
            )

            # Update viewport from the actual browser window dimensions
            # This is crucial for Browserbase/CDP connections where viewport_size may be incorrect
            # For GoogleCUAClient, this sets the viewport dimensions used for coordinate normalization
            await self._update_viewport_from_page()

            try:
                agent_result = await self.client.run_task(
                    instruction=instruction,
                    max_steps=options.max_steps or self.config.max_steps,
                    options=options,
                )
            except Exception as e:
                self.logger.error(
                    f"Exception during client.run_task: {e}", category="agent"
                )
                empty_usage = AgentUsage(
                    input_tokens=0, output_tokens=0, inference_time_ms=0
                )
                return AgentResult(
                    message=f"Error: {str(e)}",
                    completed=True,
                    actions=[],
                    usage=empty_usage,
                )

            # Update metrics if usage data is available in the result
            if agent_result.usage:
                # self.stagehand.update_metrics(
                #     AGENT_METRIC_FUNCTION_NAME,
                #     agent_result.usage.get("input_tokens", 0),
                #     agent_result.usage.get("output_tokens", 0),
                #     agent_result.usage.get("inference_time_ms", 0),
                # )
                pass  # Placeholder if metrics are to be handled differently or not at all

            self.logger.info(
                f"Agent execution finished. Success: {agent_result.completed}. Message: {agent_result.message}",
                category="agent",
            )
            self.logger.debug(
                f"Agent actions: {agent_result.actions}",
                category="agent",
            )
            return agent_result
        else:
            agent_config_payload = self.config.model_dump(
                exclude_none=True, by_alias=True
            )
            agent_config_payload["provider"] = self.provider
            payload = {
                # Use the stored config
                "agentConfig": agent_config_payload,
                "executeOptions": options.model_dump(exclude_none=True, by_alias=True),
            }

            lock = self.stagehand._get_lock_for_session()
            async with lock:
                result = await self.stagehand._execute("agentExecute", payload)

            if isinstance(result, dict):
                # Ensure all expected fields are present
                # If not present in result, use defaults from AgentExecuteResult schema
                if "success" not in result:
                    raise ValueError("Response missing required field 'success'")

                # Ensure completed is set with default if not present
                if "completed" not in result:
                    result["completed"] = False

                # Add default for message if missing
                if "message" not in result:
                    result["message"] = None

                return AgentExecuteResult(**result)
            elif result is None:
                # Handle cases where the server might return None or an empty response
                # Return a default failure result or raise an error
                return AgentExecuteResult(
                    success=False,
                    completed=False,
                    message="No result received from server",
                )
            else:
                # If the result is not a dict and not None, it's unexpected
                raise TypeError(f"Unexpected result type from server: {type(result)}")
