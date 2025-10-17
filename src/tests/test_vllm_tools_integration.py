"""
Integration tests for vLLM tools.

These tests start REAL vLLM servers with actual models and verify functionality.
NO MOCKS - all tests use actual vLLM server instances.

Requirements:
- vLLM must be installed: pip install copra-theorem-prover[os_models]
- OpenAI Python client: pip install openai (for Kimina prover tests)
- CUDA-capable GPU recommended (tests will use CPU fallback if unavailable)
- Internet connection for downloading model weights on first run

Usage:
    python src/tests/test_vllm_tools_integration.py

Test Classes:
- TestVLLMAvailability: Checks if vLLM is installed
- TestVLLMServerLifecycle: Tests server start/stop operations
- TestVLLMServerFunctionality: Tests basic API functionality with TinyLlama
- TestKiminaProverVLLM: Tests AI-MO Kimina prover models via OpenAI client
- TestVLLMServerConfigurations: Tests various server configurations

Note: Tests may take several minutes on first run due to model download.
"""

import sys
import time
import unittest
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from copra.tools.vllm_tools import has_vllm, start_server, stop_server


# Use a very lightweight model for testing
# TinyLlama is ~1.1B parameters, small enough for testing
TEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Alternative lightweight models if TinyLlama doesn't work:
# TEST_MODEL = "facebook/opt-125m"  # 125M params, very small
# TEST_MODEL = "gpt2"  # 117M params, smallest option


class TestVLLMAvailability(unittest.TestCase):
    """Test vLLM availability detection."""

    def test_has_vllm(self):
        """Test that has_vllm() correctly detects vLLM installation."""
        result = has_vllm()

        if result:
            print("✅ vLLM is installed and available")
        else:
            print("⚠️  vLLM is not installed - install with: pip install copra-theorem-prover[os_models]")

        # This test documents vLLM availability but doesn't fail
        # Other tests will be skipped if vLLM is not available
        self.assertIsInstance(result, bool)


@unittest.skipIf(not has_vllm(), "vLLM not installed - install with: pip install copra-theorem-prover[os_models]")
class TestVLLMServerLifecycle(unittest.TestCase):
    """Test vLLM server start/stop lifecycle with real server."""

    def setUp(self):
        """Set up test - no server running initially."""
        self.proc = None
        self.base_url = None
        print(f"\n{'='*60}")
        print(f"Starting test: {self._testMethodName}")
        print(f"Model: {TEST_MODEL}")
        print(f"{'='*60}")

    def tearDown(self):
        """Clean up - ensure server is stopped."""
        if self.proc:
            print(f"\n🛑 Stopping vLLM server...")
            stop_server(self.proc)
            self.proc = None
            time.sleep(2)  # Give it time to clean up

    def test_start_and_stop_server(self):
        """Test starting and stopping a real vLLM server."""
        print(f"\n🚀 Starting vLLM server with model: {TEST_MODEL}")
        print("⏳ This may take a few minutes on first run (model download)...")

        # Start server with minimal configuration
        # Use smaller max_model_len to reduce memory usage
        start_time = time.time()
        self.base_url, self.proc = start_server(
            model=TEST_MODEL,
            max_model_len=512,  # Small context for testing
            wait_seconds=300,   # Give it 5 minutes to download/load
        )
        elapsed = time.time() - start_time

        print(f"✅ Server started in {elapsed:.1f}s at {self.base_url}")

        # Verify we got a URL
        self.assertIsNotNone(self.base_url)
        self.assertIn("http://", self.base_url)

        # Verify process is running or was already running
        if self.proc:
            self.assertIsNone(self.proc.poll(), "Server process should be running")
            print(f"✅ Server process is running (PID: {self.proc.pid})")
        else:
            print(f"✅ Server was already running at {self.base_url}")

        # Test stopping the server
        if self.proc:
            print(f"\n🛑 Stopping server...")
            stop_server(self.proc)
            time.sleep(2)

            # Verify process is stopped
            self.assertIsNotNone(self.proc.poll(), "Server process should be stopped")
            print(f"✅ Server stopped successfully")
            self.proc = None

    def test_server_reuse(self):
        """Test that starting server twice reuses existing server."""
        print(f"\n🚀 Starting first vLLM server...")

        # Start first server
        start_time = time.time()
        base_url1, proc1 = start_server(
            model=TEST_MODEL,
            max_model_len=512,
            wait_seconds=300,
        )
        elapsed1 = time.time() - start_time
        print(f"✅ First server started in {elapsed1:.1f}s at {base_url1}")

        self.base_url = base_url1
        self.proc = proc1

        # Try to start second server on same port
        print(f"\n🔄 Starting second server (should reuse existing)...")
        start_time = time.time()

        # Extract port from base_url1
        port = int(base_url1.split(":")[-1].split("/")[0])

        base_url2, proc2 = start_server(
            model=TEST_MODEL,
            port=port,
            max_model_len=512,
            wait_seconds=10,  # Should be fast if reusing
        )
        elapsed2 = time.time() - start_time
        print(f"✅ Second call completed in {elapsed2:.1f}s")

        # Should get same URL
        self.assertEqual(base_url1, base_url2)

        # Should not have started new process (proc2 should be None)
        self.assertIsNone(proc2, "Should reuse existing server, not start new process")
        print(f"✅ Server was reused (no new process started)")


@unittest.skipIf(not has_vllm(), "vLLM not installed")
class TestVLLMServerFunctionality(unittest.TestCase):
    """Test vLLM server functionality with real API calls."""

    @classmethod
    def setUpClass(cls):
        """Start server once for all tests in this class."""
        print(f"\n{'='*60}")
        print(f"Setting up vLLM server for functionality tests")
        print(f"Model: {TEST_MODEL}")
        print(f"{'='*60}")

        cls.base_url, cls.proc = start_server(
            model=TEST_MODEL,
            max_model_len=512,
            wait_seconds=300,
        )

        print(f"✅ Server ready at {cls.base_url}")

    @classmethod
    def tearDownClass(cls):
        """Stop server after all tests."""
        if cls.proc:
            print(f"\n🛑 Stopping vLLM server...")
            stop_server(cls.proc)
            time.sleep(2)

    def test_server_health(self):
        """Test that server is healthy and serving the model."""
        import requests

        print(f"\n🏥 Checking server health...")

        # Check /models endpoint
        response = requests.get(
            f"{self.base_url}/models",
            headers={"Authorization": "Bearer token-abc123"},
            timeout=5.0
        )

        self.assertEqual(response.status_code, 200)
        print(f"✅ Server returned status 200")

        # Check model is in response
        data = response.json()
        self.assertIn("data", data)
        models = data["data"]
        self.assertTrue(len(models) > 0, "Should have at least one model")

        model_ids = [m.get("id") for m in models]
        self.assertIn(TEST_MODEL, model_ids)
        print(f"✅ Model {TEST_MODEL} is available")

    def test_completion_request(self):
        """Test making a real completion request to the server."""
        import requests

        print(f"\n💬 Testing completion request...")

        # Make a simple completion request
        response = requests.post(
            f"{self.base_url}/completions",
            headers={"Authorization": "Bearer token-abc123"},
            json={
                "model": TEST_MODEL,
                "prompt": "2 + 2 = ",
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=30.0
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)

        completion = data["choices"][0]["text"]
        print(f"✅ Got completion: '{completion.strip()}'")
        self.assertIsInstance(completion, str)
        self.assertTrue(len(completion) > 0)

    def test_chat_completion_request(self):
        """Test making a real chat completion request."""
        import requests

        print(f"\n💬 Testing chat completion request...")

        # Make a chat completion request
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": "Bearer token-abc123"},
            json={
                "model": TEST_MODEL,
                "messages": [
                    {"role": "user", "content": "Say 'hello' and nothing else."}
                ],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=30.0
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)

        message = data["choices"][0]["message"]["content"]
        print(f"✅ Got chat response: '{message.strip()}'")
        self.assertIsInstance(message, str)
        self.assertTrue(len(message) > 0)


@unittest.skipIf(not has_vllm(), "vLLM not installed")
class TestKiminaProverVLLM(unittest.TestCase):
    """Test Kimina prover models via vLLM with OpenAI client."""

    @classmethod
    def setUpClass(cls):
        """Start server once for all tests in this class."""
        print(f"\n{'='*60}")
        print(f"Setting up vLLM server for Kimina Prover tests")
        print(f"Model: AI-MO/Kimina-Prover-Distill-1.7B")
        print(f"{'='*60}")

        # Use the smallest Kimina model for testing (1.7B)
        # Alternative options:
        # - AI-MO/Kimina-Prover-Distill-8B (medium)
        # - AI-MO/Kimina-Prover-72B (large, requires significant resources)
        cls.kimina_model = "AI-MO/Kimina-Prover-Distill-1.7B"

        cls.base_url, cls.proc = start_server(
            model=cls.kimina_model,
            max_model_len=2048,  # Kimina models work with mathematical proofs
            wait_seconds=300,    # First run may need time to download
        )

        print(f"✅ Server ready at {cls.base_url}")

    @classmethod
    def tearDownClass(cls):
        """Stop server after all tests."""
        if cls.proc:
            print(f"\n🛑 Stopping Kimina vLLM server...")
            stop_server(cls.proc)
            time.sleep(2)

    def test_kimina_with_openai_client(self):
        """Test Kimina prover using OpenAI Python client."""
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI Python client not installed - install with: pip install openai")

        print(f"\n🤖 Testing Kimina prover with OpenAI client...")

        # Create OpenAI client pointing to vLLM server
        client = OpenAI(
            api_key="token-abc123",
            base_url=self.base_url,
        )

        # Test a simple mathematical problem
        # Kimina is designed for theorem proving, so we'll test with a math proof request
        prompt = """Prove that the sum of two even numbers is even.

Let's denote two even numbers as 2m and 2n, where m and n are integers."""

        print(f"📝 Prompt: {prompt[:100]}...")

        # Make completion request
        start_time = time.time()
        completion = client.completions.create(
            model=self.kimina_model,
            prompt=prompt,
            max_tokens=256,
            temperature=0.0,  # Deterministic for testing
        )
        elapsed = time.time() - start_time

        # Verify response
        self.assertIsNotNone(completion)
        self.assertTrue(len(completion.choices) > 0)

        response_text = completion.choices[0].text
        print(f"✅ Got response in {elapsed:.2f}s: '{response_text.strip()[:200]}...'")

        self.assertIsInstance(response_text, str)
        self.assertTrue(len(response_text) > 0)

    def test_kimina_chat_completion_openai_client(self):
        """Test Kimina prover using OpenAI chat completions API."""
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI Python client not installed - install with: pip install openai")

        print(f"\n💬 Testing Kimina prover with chat completions API...")

        # Create OpenAI client pointing to vLLM server
        client = OpenAI(
            api_key="token-abc123",
            base_url=self.base_url,
        )

        # Test with a mathematical theorem proving task
        messages = [
            {
                "role": "system",
                "content": "You are a mathematical theorem prover. Provide clear, rigorous proofs."
            },
            {
                "role": "user",
                "content": "Prove that for any integer n, n^2 is non-negative."
            }
        ]

        print(f"📝 Testing chat completion with theorem proving task...")

        # Make chat completion request
        start_time = time.time()
        response = client.chat.completions.create(
            model=self.kimina_model,
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )
        elapsed = time.time() - start_time

        # Verify response
        self.assertIsNotNone(response)
        self.assertTrue(len(response.choices) > 0)

        message_content = response.choices[0].message.content
        print(f"✅ Got chat response in {elapsed:.2f}s: '{message_content.strip()[:200]}...'")

        self.assertIsInstance(message_content, str)
        self.assertTrue(len(message_content) > 0)

    def test_kimina_lean4_proof_format(self):
        """Test Kimina prover with Lean 4 proof format (its native format)."""
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI Python client not installed - install with: pip install openai")

        print(f"\n🎯 Testing Kimina with Lean 4 proof format...")

        client = OpenAI(
            api_key="token-abc123",
            base_url=self.base_url,
        )

        # Kimina is specifically designed for Lean 4 theorem proving
        lean4_prompt = """theorem add_comm (a b : Nat) : a + b = b + a :="""

        print(f"📝 Lean 4 theorem: {lean4_prompt}")

        start_time = time.time()
        completion = client.completions.create(
            model=self.kimina_model,
            prompt=lean4_prompt,
            max_tokens=512,  # Proofs can be longer
            temperature=0.2,  # Slight randomness for proof exploration
        )
        elapsed = time.time() - start_time

        response_text = completion.choices[0].text
        print(f"✅ Generated Lean 4 proof in {elapsed:.2f}s")
        print(f"📄 Proof snippet: '{response_text.strip()[:300]}...'")

        self.assertIsInstance(response_text, str)
        self.assertTrue(len(response_text) > 0)


@unittest.skipIf(not has_vllm(), "vLLM not installed")
class TestVLLMServerConfigurations(unittest.TestCase):
    """Test different vLLM server configurations."""

    def tearDown(self):
        """Clean up any servers started during test."""
        if hasattr(self, 'proc') and self.proc:
            stop_server(self.proc)
            time.sleep(2)

    def test_custom_port(self):
        """Test starting server on custom port."""
        print(f"\n🔧 Testing custom port configuration...")

        # Use a specific high port
        custom_port = 8001

        base_url, self.proc = start_server(
            model=TEST_MODEL,
            port=custom_port,
            max_model_len=512,
            wait_seconds=300,
        )

        self.assertIn(f":{custom_port}", base_url)
        print(f"✅ Server started on custom port {custom_port}")

    def test_custom_api_key(self):
        """Test server with custom API key."""
        import requests

        print(f"\n🔑 Testing custom API key...")

        custom_key = "test-key-12345"

        base_url, self.proc = start_server(
            model=TEST_MODEL,
            api_key=custom_key,
            max_model_len=512,
            wait_seconds=300,
        )

        # Try with correct key
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {custom_key}"},
            timeout=5.0
        )
        self.assertEqual(response.status_code, 200)
        print(f"✅ Custom API key works")

        # Try with wrong key (should fail)
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": "Bearer wrong-key"},
            timeout=5.0
        )
        self.assertNotEqual(response.status_code, 200)
        print(f"✅ Wrong API key rejected")


def run_tests():
    """Run all tests and print summary."""
    print("\n" + "="*70)
    print("vLLM Tools Integration Tests")
    print("="*70)
    print(f"\nTest Model: {TEST_MODEL}")
    print("\n⚠️  WARNING: These tests make REAL API calls to vLLM servers")
    print("⚠️  First run may take several minutes (model download)")
    print("⚠️  Requires GPU for best performance (CPU fallback available)")
    print("\n" + "="*70 + "\n")

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVLLMAvailability))
    suite.addTests(loader.loadTestsFromTestCase(TestVLLMServerLifecycle))
    suite.addTests(loader.loadTestsFromTestCase(TestVLLMServerFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestKiminaProverVLLM))
    suite.addTests(loader.loadTestsFromTestCase(TestVLLMServerConfigurations))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")

    print("="*70 + "\n")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
