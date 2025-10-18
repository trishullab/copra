#!/usr/bin/env python3

"""
Tests for SimpleStringPolicyPrompter.

This test file demonstrates the usage of SimpleStringPolicyPrompter
and can be run manually to verify functionality with actual LLM calls.

Note: These tests require valid API credentials and will make real API calls.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from copra.agent.simple_string_policy_prompter import SimpleStringPolicyPrompter


def test_basic_run_prompt():
    """Test basic run_prompt() functionality."""
    print("\n" + "="*60)
    print("Test 1: Basic run_prompt()")
    print("="*60)

    prompter = SimpleStringPolicyPrompter(
        gpt_model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens_per_action=100,
        system_prompt="You are a helpful assistant that gives concise answers.",
        stop_tokens=["END", "\n\n"]
    )

    with prompter:
        message = "What is the capital of France?"
        print(f"Q: {message}")
        responses = prompter.run_prompt(message)
        print(f"A: {responses[0]['content']}")

        assert len(responses) > 0, "Should return at least one response"
        assert 'content' in responses[0], "Response should have 'content' key"

    print("✅ Test passed!")


def test_chat_interface():
    """Test convenient chat() interface."""
    print("\n" + "="*60)
    print("Test 2: Chat interface")
    print("="*60)

    prompter = SimpleStringPolicyPrompter(
        gpt_model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens_per_action=100,
        system_prompt="You are a helpful assistant that gives concise answers.",
        stop_tokens=["END", "\n\n"]
    )

    with prompter:
        question = "What is 2 + 2?"
        print(f"Q: {question}")
        response = prompter.chat(question)
        print(f"A: {response}")

        assert isinstance(response, str), "chat() should return a string"
        assert len(response) > 0, "Response should not be empty"

    print("✅ Test passed!")


def test_with_history():
    """Test conversation history management."""
    print("\n" + "="*60)
    print("Test 3: Conversation with history")
    print("="*60)

    prompter = SimpleStringPolicyPrompter(
        gpt_model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens_per_action=100,
        max_history_messages=4,  # Keep up to 4 messages in history
        system_prompt="You are a helpful assistant that gives concise answers.",
        stop_tokens=["END", "\n\n"]
    )

    with prompter:
        # First message - establish context
        msg1 = "My name is Alice"
        print(f"User: {msg1}")
        response1 = prompter.chat(msg1)
        print(f"Assistant: {response1}")

        # Second message - reference previous context
        msg2 = "What is my name?"
        print(f"\nUser: {msg2}")
        response2 = prompter.chat(msg2)
        print(f"Assistant: {response2}")

        assert isinstance(response2, str), "chat() should return a string"
        # Note: We can't assert the content without mocking, but we verify it works

    print("✅ Test passed!")


def test_long_message_truncation():
    """Test that long messages are properly truncated."""
    print("\n" + "="*60)
    print("Test 4: Long message truncation")
    print("="*60)

    prompter = SimpleStringPolicyPrompter(
        gpt_model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens_per_action=50,
        system_prompt="You are a helpful assistant.",
        stop_tokens=["END"]
    )

    with prompter:
        # Create a very long message
        long_message = "Tell me about quantum computing. " * 500  # Very long!
        print(f"Message length: {len(long_message)} characters")

        # This should not fail even though the message is very long
        response = prompter.chat(long_message)
        print(f"Response: {response[:100]}...")  # Print first 100 chars

        assert isinstance(response, str), "Should still return a string"

    print("✅ Test passed!")


def test_token_management():
    """Test that token management works correctly."""
    print("\n" + "="*60)
    print("Test 5: Token management")
    print("="*60)

    prompter = SimpleStringPolicyPrompter(
        gpt_model_name="gpt-4o-mini",
        temperature=0.0,  # Deterministic
        max_tokens_per_action=20,  # Small limit
        system_prompt="Be very brief.",
        stop_tokens=["END"]
    )

    with prompter:
        message = "Count to 100"
        print(f"Q: {message} (with max_tokens=20)")
        response = prompter.chat(message)
        print(f"A: {response}")

        # The response should be truncated due to token limit
        assert isinstance(response, str), "Should return a string"

    print("✅ Test passed!")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing SimpleStringPolicyPrompter")
    print("="*60)
    print("\nNote: These tests require valid OpenAI API credentials")
    print("and will make real API calls (costs may apply).\n")

    try:
        test_basic_run_prompt()
        test_chat_interface()
        test_with_history()
        test_long_message_truncation()
        test_token_management()

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
