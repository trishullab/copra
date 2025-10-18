#!/usr/bin/env python3

"""
SimpleStringPolicyPrompter: A concrete implementation that accepts plain string messages.

This is a lightweight prompter for use cases where you just want to send string messages
to the LLM without complex grammar parsing or retrieval mechanisms.

Example usage:
    prompter = SimpleStringPolicyPrompter(
        gpt_model_name="gpt-4o",
        temperature=0.7,
        max_tokens_per_action=100,
        stop_tokens=["END"]
    )

    with prompter:
        responses = prompter.run_prompt("Explain quantum computing in simple terms")
        print(responses[0]['content'])
"""

import typing
import logging
from copra.agent.simple_policy_prompter import SimplePolicyPrompter


class SimpleStringPolicyPrompter(SimplePolicyPrompter):
    """
    A simple policy prompter that works with plain string messages.

    This class provides a straightforward interface for sending string prompts
    to LLMs without requiring complex formatting or domain-specific grammars.
    """

    def __init__(
        self,
        gpt_model_name: str = "gpt-3.5-turbo",
        secret_filepath: typing.Optional[str] = None,
        temperature: float = 0.25,
        num_sequences: int = 1,
        max_tokens_per_action: int = 50,
        max_history_messages: int = 0,
        logger: typing.Optional[logging.Logger] = None,
        model_params: typing.Optional[typing.Dict[str, typing.Any]] = None,
        system_prompt: typing.Optional[str] = None,
        stop_tokens: typing.Optional[typing.List[str]] = None
    ):
        """
        Initialize the simple string policy prompter.

        Args:
            gpt_model_name: Model name (e.g., "gpt-3.5-turbo", "gpt-4o")
            secret_filepath: Path to API key file (optional)
            temperature: Sampling temperature (0.0 to 1.0)
            num_sequences: Number of sequences to generate
            max_tokens_per_action: Maximum tokens to generate per action
            max_history_messages: Maximum messages to keep in history (0 = no history)
            logger: Logger instance (creates default if None)
            model_params: Additional model parameters
            system_prompt: Optional system prompt string
            stop_tokens: List of stop sequences (default: ["[END]"])
        """
        super().__init__(
            gpt_model_name=gpt_model_name,
            secret_filepath=secret_filepath,
            temperature=temperature,
            num_sequences=num_sequences,
            max_tokens_per_action=max_tokens_per_action,
            max_history_messages=max_history_messages,
            logger=logger,
            model_params=model_params
        )

        # Set up system messages from string
        if system_prompt:
            self.system_messages = [{"role": "system", "content": system_prompt}]
            self.system_token_count = self._gpt_access.num_tokens_from_messages(self.system_messages)
        else:
            self.system_messages = []
            self.system_token_count = 0

        # Set stop tokens
        self._stop_tokens = stop_tokens if stop_tokens is not None else ["[END]"]

    def run_prompt(self, message: str) -> typing.List[typing.Dict[str, str]]:
        """
        Send a plain string message to the LLM and get responses.

        Args:
            message: Plain text message to send

        Returns:
            List of response dictionaries with 'role' and 'content'

        Example:
            >>> responses = prompter.run_prompt("What is 2 + 2?")
            >>> print(responses[0]['content'])
            "2 + 2 equals 4."
        """
        return self.run_prompt_simple(message, stop_tokens=self._stop_tokens)

    def parse_response(self, response: typing.List[typing.Dict[str, str]]) -> str:
        """
        Parse LLM response to extract content.

        Args:
            response: List of response dictionaries from run_prompt()

        Returns:
            Content string from the first response
        """
        if response and len(response) > 0:
            return response[0].get('content', '')
        return ''

    def chat(self, message: str) -> str:
        """
        Convenient chat interface - send message and get string response.

        Args:
            message: User message string

        Returns:
            Assistant response string

        Example:
            >>> response = prompter.chat("Hello!")
            >>> print(response)
            "Hello! How can I help you today?"
        """
        responses = self.run_prompt(message)
        return self.parse_response(responses)
