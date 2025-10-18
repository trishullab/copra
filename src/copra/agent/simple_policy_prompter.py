#!/usr/bin/env python3

"""
SimplePolicyPrompter: Base class for LLM-based policy prompting.

This class provides common functionality for interacting with LLMs (OpenAI GPT, Llama, etc.)
including token management, rate limiting, message history, and retry logic.
"""

import time
import typing
import logging
from copra.agent.rate_limiter import RateLimiter
from copra.gpts.gpt_access import GptAccess
from copra.gpts.llama_access import LlamaAccess, ServiceDownError
from copra.prompt_generator.prompter import PolicyPrompter
from copra.tools.misc import model_supports_openai_api


class SimplePolicyPrompter(PolicyPrompter):
    """
    Base class for policy prompters that interact with LLMs.

    Provides common functionality:
    - LLM access (GptAccess or LlamaAccess)
    - Token counting and management
    - Rate limiting with automatic throttling
    - Message history management
    - Retry logic with exponential backoff
    - Context manager support
    """

    def __init__(
        self,
        gpt_model_name: str = "gpt-3.5-turbo",
        secret_filepath: str = None,
        temperature: float = 0.25,
        num_sequences: int = 1,
        max_tokens_per_action: int = 50,
        max_history_messages: int = 0,
        logger: typing.Optional[logging.Logger] = None,
        model_params: typing.Optional[typing.Dict[str, typing.Any]] = None
    ):
        """
        Initialize the simple policy prompter.

        Args:
            gpt_model_name: Name of the model (e.g., "gpt-3.5-turbo", "gpt-4o")
            secret_filepath: Path to API key file (optional, uses default if None)
            temperature: Sampling temperature (0.0 to 1.0)
            num_sequences: Number of sequences to generate
            max_tokens_per_action: Maximum tokens to generate per action
            max_history_messages: Maximum number of history messages to keep (0 = no history)
            logger: Logger instance (creates default if None)
            model_params: Additional model parameters
        """
        super().__init__()

        self.model_name = gpt_model_name
        self.temperature = temperature
        self.num_sequences = num_sequences
        self._max_tokens_per_action = max_tokens_per_action
        self._max_history_messages = max_history_messages

        # Filter out server-only parameters from model_params
        # These are used for vLLM/server initialization, not for API calls
        server_only_params = {'max_model_len', 'gpu_memory_utilization', 'tensor_parallel_size'}
        if model_params is not None:
            self._model_params = {k: v for k, v in model_params.items() if k not in server_only_params}
        else:
            self._model_params = {}

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Initialize LLM access (GptAccess or LlamaAccess)
        # Note: vLLM models (with "vllm:" prefix) are handled by GptAccess
        use_defensive_parsing = not model_supports_openai_api(gpt_model_name)
        if not model_supports_openai_api(gpt_model_name):
            self._gpt_access = LlamaAccess(gpt_model_name)
        else:
            self._gpt_access = GptAccess(secret_filepath=secret_filepath, model_name=gpt_model_name)

        # Get model configuration
        # For vLLM models, use the generic "vllm" key in model_info
        from copra.tools.misc import is_vllm_model
        model_info_key = "vllm" if is_vllm_model(gpt_model_name) else gpt_model_name
        self._token_limit_per_min = GptAccess.gpt_model_info[model_info_key]["token_limit_per_min"]
        self._request_limit_per_min = GptAccess.gpt_model_info[model_info_key]["request_limit_per_min"]
        self._max_token_per_prompt = GptAccess.gpt_model_info[model_info_key]["max_token_per_prompt"]

        # Initialize rate limiter
        self._rate_limiter = RateLimiter(self._token_limit_per_min, self._request_limit_per_min)

        # Initialize message history
        self._message_history: typing.List[typing.Dict[str, str]] = []
        self._message_history_token_count: typing.List[int] = []
        self._history_token_count = 0

        # System messages (to be set by subclasses)
        self.system_messages: typing.List[typing.Dict[str, str]] = []
        self.system_token_count = 0

        # Tracking
        self._num_api_calls = 0
        self.last_message_has_error = False

    def __enter__(self):
        """Context manager entry - initialize LLM service if needed."""
        if isinstance(self._gpt_access, LlamaAccess):
            self._gpt_access.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - cleanup LLM service if needed."""
        if isinstance(self._gpt_access, LlamaAccess):
            self._gpt_access.__exit__(exc_type, exc_value, traceback)

    def add_to_history(self, message: typing.Dict[str, str]):
        """
        Add a message to the history.

        Args:
            message: Message dictionary with 'role' and 'content'
        """
        message_token_count = self._gpt_access.num_tokens_from_messages([message])
        self._message_history.append(message)
        self._message_history_token_count.append(message_token_count)
        self._history_token_count += message_token_count

    def reset_last_message(self, message: typing.Dict[str, str]):
        """
        Replace the last message in history.

        Args:
            message: New message to replace the last one
        """
        if len(self._message_history) > 0:
            self._history_token_count -= self._message_history_token_count[-1]
            self._message_history.pop()
            self._message_history_token_count.pop()
        self.add_to_history(message)

    def clear_history(self):
        """Clear all message history."""
        self._message_history = []
        self._message_history_token_count = []
        self._history_token_count = 0

    def _constrain_tokens_in_history(
        self,
        prompt_message: typing.Dict[str, str],
        custom_example_system_messages: typing.List[typing.Dict[str, str]],
        custom_system_message_count: int,
        prompt_token_count: int,
        max_tokens_per_action: int
    ) -> typing.Tuple[typing.List[typing.Dict[str, str]], int]:
        """
        Constrain the token count by removing old history messages if necessary.

        Args:
            prompt_message: The current prompt message
            custom_example_system_messages: Additional system messages (e.g., examples)
            custom_system_message_count: Token count for custom system messages
            prompt_token_count: Token count for the prompt message
            max_tokens_per_action: Maximum tokens to reserve for generation

        Returns:
            Tuple of (messages list, total token count)
        """
        # Determine which history messages to keep
        if len(self._message_history) >= self._max_history_messages:
            if not self.last_message_has_error:
                history_idx = len(self._message_history) - self._max_history_messages
            else:
                history_idx = 0
        else:
            history_idx = 0

        if history_idx < len(self._message_history):
            # Calculate total token count
            total_token_count = (
                self.system_token_count +
                self._history_token_count +
                prompt_token_count +
                custom_system_message_count
            )
            max_token_per_prompt = min(
                self._max_token_per_prompt,
                self._max_token_per_prompt - max_tokens_per_action
            )
            assert max_token_per_prompt > 0, (
                "Max token per prompt must be greater than 0, please decrease max_tokens_per_action"
            )

            # Remove old messages if we exceed the token limit
            tokens_shredded = False
            remove_cnt = 0
            history_count = self._history_token_count
            while total_token_count >= max_token_per_prompt and history_idx < len(self._message_history):
                self.logger.warning(
                    f"Tokens exceeded removing history at index {history_idx}: "
                    f"{total_token_count} >= {max_token_per_prompt}"
                )
                history_count -= self._message_history_token_count[history_idx]
                total_token_count = (
                    self.system_token_count +
                    history_count +
                    prompt_token_count +
                    custom_system_message_count
                )
                history_idx += 1
                tokens_shredded = True
                remove_cnt += 1

            # Ensure we remove messages in pairs (to keep conversation structure)
            if remove_cnt % 2 == 1 and history_idx < len(self._message_history):
                history_count -= self._message_history_token_count[history_idx]
                total_token_count = (
                    self.system_token_count +
                    history_count +
                    prompt_token_count +
                    custom_system_message_count
                )
                history_idx += 1

            if tokens_shredded:
                self.logger.warning(
                    f"Shredded tokens from history. New total token count: {total_token_count}, "
                    f"max token per prompt: {max_token_per_prompt}, "
                    f"history token count: {self._history_token_count}, "
                    f"prompt token count: {prompt_token_count}"
                )

            if total_token_count >= max_token_per_prompt:
                self.logger.warning(
                    f"Total token count {total_token_count} is still greater than "
                    f"max token per prompt {max_token_per_prompt}."
                )
        else:
            total_token_count = self.system_token_count + prompt_token_count + custom_system_message_count

        # Remove old messages from history
        if history_idx > 0:
            for idx in range(min(history_idx, len(self._message_history))):
                self._history_token_count -= self._message_history_token_count[idx]

        self._message_history = self._message_history[history_idx:]
        self._message_history_token_count = self._message_history_token_count[history_idx:]

        # Add the new prompt message to history
        self._message_history.append(prompt_message)
        self._message_history_token_count.append(prompt_token_count)
        self._history_token_count += prompt_token_count + custom_system_message_count

        # Construct the full message list
        messages = self.system_messages + custom_example_system_messages + self._message_history

        assert total_token_count + max_tokens_per_action <= self._max_token_per_prompt, (
            f"Total token count {total_token_count} + max tokens per action {max_tokens_per_action} "
            f"is greater than max token per prompt {self._max_token_per_prompt}"
        )

        return messages, total_token_count

    def _throttle_if_needed(self, total_token_count: int):
        """
        Check rate limits and sleep if necessary.

        Args:
            total_token_count: Token count for the current request
        """
        has_hit_rate_limit = self._rate_limiter.check(total_token_count)
        was_throttled = False
        while not has_hit_rate_limit:
            current_time = time.time()
            time_to_sleep = max(1, 60 - (current_time - self._rate_limiter._last_request_time))
            self.logger.info(
                f"Rate limit reached. Sleeping for {time_to_sleep} seconds. "
                f"Rate limiter info: {self._rate_limiter}"
            )
            time.sleep(time_to_sleep)
            has_hit_rate_limit = self._rate_limiter.check(total_token_count)
            was_throttled = True

        if was_throttled:
            self.logger.info("Rate limit was hit. So the request was throttled.")
            self._rate_limiter.reset()
            self.logger.info("Rate limit reset now.")

    def _get_prompt_message_from_string(
        self,
        message: str,
        max_tokens_in_prompt: int,
        role: str = "user"
    ) -> typing.Tuple[typing.Dict[str, str], int]:
        """
        Convert a plain string message to a formatted prompt message with token constraints.

        This is a simpler version that takes a raw string and ensures it fits within
        the token budget by truncating if necessary.

        Args:
            message: The raw message string
            max_tokens_in_prompt: Maximum tokens allowed for this prompt
            role: Message role (default: "user")

        Returns:
            Tuple of (formatted_message_dict, token_count)
        """
        assert max_tokens_in_prompt > 0, (
            "Max token per prompt must be greater than 0, please decrease max_tokens_per_action"
        )

        # Initial estimation: ~4 characters per token
        characters_per_token = 4.0
        full_prompt_message = message
        prompt_char_cnt = len(full_prompt_message)

        # Create OpenAI message format
        full_prompt_message_dict = {"role": role, "content": full_prompt_message}
        prompt_token_count = self._gpt_access.num_tokens_from_messages([full_prompt_message_dict])

        # Refine character/token ratio based on actual count
        if prompt_token_count > 0:
            characters_per_token = prompt_char_cnt / prompt_token_count

        decrement_factor = 0.1
        characters_per_token -= decrement_factor
        retries = 50
        prompt_message = full_prompt_message_dict
        prompt_messages = [full_prompt_message_dict]

        # If already under limit, return immediately
        if prompt_token_count <= max_tokens_in_prompt:
            return prompt_message, prompt_token_count

        # Iteratively truncate message to fit token budget
        while prompt_token_count > max_tokens_in_prompt and retries > 0 and characters_per_token > 0:
            max_chars_in_prompt = int(max_tokens_in_prompt * characters_per_token)
            truncated_message = message[:max_chars_in_prompt]
            prompt_char_cnt = len(truncated_message)

            # Create message dict
            prompt_message = {"role": role, "content": truncated_message}
            prompt_messages = [prompt_message]
            prompt_token_count = self._gpt_access.num_tokens_from_messages(prompt_messages)

            retries -= 1
            characters_per_token -= decrement_factor

            if prompt_token_count > max_tokens_in_prompt:
                self.logger.warning(
                    f"Prompt token count {prompt_token_count} is greater than max token per prompt "
                    f"{max_tokens_in_prompt}. Retrying with {characters_per_token} characters per token."
                )

            assert prompt_char_cnt > 0, (
                f"Prompt message is empty. Please decrease max_tokens_per_action. "
                f"Current value: {self._max_tokens_per_action}"
            )

        # Final check
        prompt_token_count = self._gpt_access.num_tokens_from_messages(prompt_messages)
        assert prompt_token_count <= max_tokens_in_prompt, (
            f"Prompt token count {prompt_token_count} is greater than max token per prompt "
            f"{max_tokens_in_prompt}"
        )

        return prompt_message, prompt_token_count

    def _constrain_tokens_in_history_simple(
        self,
        prompt_message: typing.Dict[str, str],
        prompt_token_count: int,
        max_tokens_per_action: int
    ) -> typing.Tuple[typing.List[typing.Dict[str, str]], int]:
        """
        Simpler version of _constrain_tokens_in_history without custom system messages.

        Args:
            prompt_message: The current prompt message
            prompt_token_count: Token count for the prompt message
            max_tokens_per_action: Maximum tokens to reserve for generation

        Returns:
            Tuple of (messages list, total token count)
        """
        return self._constrain_tokens_in_history(
            prompt_message=prompt_message,
            custom_example_system_messages=[],
            custom_system_message_count=0,
            prompt_token_count=prompt_token_count,
            max_tokens_per_action=max_tokens_per_action
        )

    def run_prompt_simple(
        self,
        message: str,
        stop_tokens: typing.Optional[typing.List[str]] = None
    ) -> typing.List[typing.Dict[str, str]]:
        """
        Simplified run_prompt that takes a plain string message.

        This is a convenience method for simpler use cases where you just want to
        send a string message without complex formatting or custom system messages.

        Args:
            message: Plain text message to send
            stop_tokens: Optional list of stop sequences (defaults to ["[END]"])

        Returns:
            List of response messages from the LLM

        Raises:
            Exception: If all retries fail
            ServiceDownError: If the LLM service is down
        """
        if stop_tokens is None:
            stop_tokens = ["[END]"]

        # Calculate available token budget
        max_tokens_in_prompt = (
            self._max_token_per_prompt -
            self.system_token_count -
            self._max_tokens_per_action
        )

        # Convert message to formatted prompt
        prompt_message, prompt_token_count = self._get_prompt_message_from_string(
            message, max_tokens_in_prompt
        )

        # Constrain history
        messages, total_token_count = self._constrain_tokens_in_history_simple(
            prompt_message,
            prompt_token_count,
            self._max_tokens_per_action
        )

        # Use the standard retry logic
        responses, usage = self.complete_chat_with_retry(
            messages=messages,
            total_token_count=total_token_count,
            max_tokens_per_action=self._max_tokens_per_action,
            stop=stop_tokens
        )

        return responses

    def complete_chat_with_retry(
        self,
        messages: typing.List[typing.Dict[str, str]],
        total_token_count: int,
        max_tokens_per_action: int,
        temperature: typing.Optional[float] = None,
        stop: typing.List[str] = ["[END]"],
        retries: int = 6,
        time_to_sleep: float = 60,
        exp_factor: float = 1.06,
        tokens_factor: float = 1.75,
        temp_factor: float = 0.025,
        max_temp: float = 0.4
    ) -> typing.Tuple[typing.List[typing.Dict[str, str]], typing.Dict[str, typing.Any]]:
        """
        Complete a chat request with automatic retry and token adjustment.

        Args:
            messages: List of message dictionaries
            total_token_count: Total input token count
            max_tokens_per_action: Maximum tokens to generate
            temperature: Sampling temperature (uses self.temperature if None)
            stop: Stop sequences
            retries: Maximum number of retries
            time_to_sleep: Initial sleep time on error (seconds)
            exp_factor: Exponential backoff factor for sleep time
            tokens_factor: Multiplier for increasing token limit on incomplete responses
            temp_factor: Temperature increase factor (currently unused)
            max_temp: Maximum temperature (currently unused)

        Returns:
            Tuple of (responses, usage_dict)

        Raises:
            Exception: If all retries fail
            ServiceDownError: If the LLM service is down
        """
        if temperature is None:
            temperature = self.temperature

        success = False
        tokens_to_generate = max_tokens_per_action
        upper_bound = 10 * max_tokens_per_action
        responses = None

        while not success and retries > 0:
            try:
                self._throttle_if_needed(total_token_count)
                self.logger.info(
                    f"Requesting {tokens_to_generate} tokens to generate, "
                    f"{total_token_count} tokens in input."
                )

                request_start_time = time.time()
                if len(self._model_params) > 0:
                    responses, usage = self._gpt_access.complete_chat(
                        messages,
                        n=self.num_sequences,
                        temperature=temperature,
                        max_tokens=tokens_to_generate,
                        stop=stop,
                        **self._model_params
                    )
                else:
                    responses, usage = self._gpt_access.complete_chat(
                        messages,
                        n=self.num_sequences,
                        temperature=temperature,
                        max_tokens=tokens_to_generate,
                        stop=stop
                    )
                request_end_time = time.time()

                time_taken = request_end_time - request_start_time
                approx_output_tokens = usage["total_tokens"] - total_token_count
                self.logger.info(
                    f"Request took {time_taken} seconds. Used {usage['total_tokens']} tokens. "
                    f"Used {usage['completion_tokens']} completion tokens. "
                    f"Approx. output {approx_output_tokens} tokens."
                )

                reason = usage["reason"]
                self._rate_limiter.update(usage["total_tokens"], request_start_time, request_end_time)

                # Check if response was complete
                success = reason != "length" or tokens_to_generate >= upper_bound
                if not success:
                    tokens_to_generate = min(int(tokens_to_generate * tokens_factor), upper_bound)
                    self.logger.info(
                        f"Retrying with {tokens_to_generate} tokens. "
                        f"Earlier response was not complete for reason: {reason}. "
                        f"Used {usage['completion_tokens']} completion tokens."
                    )
                    self.logger.info(f"Incomplete Response messages: \n{responses}")
                else:
                    if tokens_to_generate >= upper_bound:
                        self.logger.warning(
                            f"Retried {retries} times but still got an incomplete response. "
                            f"Reason: {reason}."
                        )
                        self.logger.info(f"Maxed out response: \n{responses}")
                    else:
                        self.logger.debug(f"Got a valid response. Reason: \n{reason}")
                        self.logger.debug(f"Response messages: \n{responses}")

                self._num_api_calls += 1

            except ServiceDownError as e:
                self.logger.info("Got a service down error. Will give up until the docker container is restarted.")
                self.logger.exception(e)
                raise
            except Exception as e:
                self.logger.info("Got an unknown exception. Retrying.")
                self.logger.exception(e)
                time.sleep(time_to_sleep)
                responses = []
                usage = {}
                time_to_sleep *= exp_factor  # Exponential backoff

            retries -= 1

        if not success and responses is None:
            raise Exception(f"Failed to get valid response after {retries} tries")

        return responses, usage

    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        """
        Get efficiency metrics.

        Returns:
            Dictionary with efficiency information
        """
        return {
            "api_calls": self._num_api_calls
        }

    # Abstract methods to be implemented by subclasses
    def run_prompt(self, request: typing.Any) -> typing.Any:
        """
        Execute a prompt request.

        Args:
            request: Request object (format depends on subclass)

        Returns:
            Response from the LLM
        """
        raise NotImplementedError("Subclasses must implement run_prompt()")

    def parse_response(self, response: typing.Any) -> typing.Any:
        """
        Parse the LLM response.

        Args:
            response: Response from the LLM

        Returns:
            Parsed response (format depends on subclass)
        """
        raise NotImplementedError("Subclasses must implement parse_response()")