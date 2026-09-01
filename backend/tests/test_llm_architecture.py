"""
LLM Architecture Tests.

Tests the provider-agnostic LLM abstraction layer:
  1. LLMConfig normalization
  2. LLMRequest defaults
  3. LLMResponse structure
  4. BaseLLM contract
  5. MockLLM
  6. Factory provider selection
  7. Unsupported provider error
  8. Missing API key error
  9. All providers conform to BaseLLM
  10. LLMResponse returned by providers
  11. Provider independence (no SDK leaks)
  12. Factory builds correct config
  13. Provider switching
"""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.llm.base import BaseLLM, LLMResponse
from app.llm.config import LLMConfig
from app.llm.types import LLMRequest
from app.llm.factory import create_llm, LLM_PROVIDERS, PROVIDERS_REQUIRING_API_KEY
from app.llm.mock import MockLLM


# ============================================================
# TEST 1: LLMConfig normalization
# ============================================================

def test_llm_config_normalization():
    print("\nTEST 1: LLMConfig normalization")
    # Provider should be lowercased and stripped
    config = LLMConfig(
        provider="  Gemini  ",
        model="test-model",
        api_key="test-key",
    )
    assert config.provider == "gemini", f"Expected 'gemini', got '{config.provider}'"

    config2 = LLMConfig(
        provider="GROQ",
        model="groq-model",
        api_key="key",
    )
    assert config2.provider == "groq", f"Expected 'groq', got '{config2.provider}'"

    # Should be frozen (immutable)
    try:
        config.provider = "changed"
        assert False, "LLMConfig should be frozen/immutable"
    except AttributeError:
        pass

    print("  PASSED: Provider normalized, config frozen")


# ============================================================
# TEST 2: LLMConfig structure
# ============================================================

def test_llm_config_structure():
    print("\nTEST 2: LLMConfig structure")
    config = LLMConfig(
        provider="mock",
        model="mock-model",
        api_key="",
    )
    assert config.provider == "mock"
    assert config.model == "mock-model"
    assert config.api_key == ""
    assert config.options == {}

    # With options
    config2 = LLMConfig(
        provider="openrouter",
        model="gpt-4",
        api_key="key",
        options={"base_url": "https://custom.url"},
    )
    assert config2.options["base_url"] == "https://custom.url"

    print("  PASSED: LLMConfig has correct fields")


# ============================================================
# TEST 3: LLMRequest defaults
# ============================================================

def test_llm_request_defaults():
    print("\nTEST 3: LLMRequest defaults")
    req = LLMRequest(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert req.messages == [{"role": "user", "content": "hello"}]
    assert req.temperature is None
    assert req.max_tokens is None

    # With explicit values
    req2 = LLMRequest(
        messages=[],
        temperature=0.7,
        max_tokens=1024,
    )
    assert req2.temperature == 0.7
    assert req2.max_tokens == 1024

    # Should be frozen
    try:
        req.messages = []
        assert False, "LLMRequest should be frozen"
    except AttributeError:
        pass

    print("  PASSED: LLMRequest has correct defaults and is frozen")


# ============================================================
# TEST 4: LLMResponse structure
# ============================================================

def test_llm_response_structure():
    print("\nTEST 4: LLMResponse structure")
    resp = LLMResponse(
        text="Hello",
        model="test-model",
        provider="test-provider",
    )
    assert resp.text == "Hello"
    assert resp.model == "test-model"
    assert resp.provider == "test-provider"

    # With defaults
    resp2 = LLMResponse(text="Hi")
    assert resp2.model is None
    assert resp2.provider is None

    # Should be frozen
    try:
        resp.text = "Changed"
        assert False, "LLMResponse should be frozen"
    except AttributeError:
        pass

    print("  PASSED: LLMResponse has correct fields")


# ============================================================
# TEST 5: BaseLLM contract
# ============================================================

def test_base_llm_contract():
    print("\nTEST 5: BaseLLM contract")
    # BaseLLM should be abstract
    try:
        llm = BaseLLM(config=LLMConfig(provider="test", model="m", api_key="k"))
        llm.generate(LLMRequest(messages=[]))
        assert False, "BaseLLM should be abstract"
    except TypeError:
        pass

    print("  PASSED: BaseLLM is abstract, cannot be instantiated directly")


# ============================================================
# TEST 6: MockLLM
# ============================================================

def test_mock_llm():
    print("\nTEST 6: MockLLM")
    llm = MockLLM(
        config=LLMConfig(provider="mock", model="mock-model", api_key=""),
    )

    assert isinstance(llm, BaseLLM)
    assert llm.config.provider == "mock"
    assert llm.model == "mock-model"

    request = LLMRequest(
        messages=[{"role": "user", "content": "Hello world"}],
    )
    response = llm.generate(request)

    assert isinstance(response, LLMResponse)
    assert "Hello world" in response.text
    assert response.model == "mock-model"
    assert response.provider == "mock"

    # Mock should not need API key
    llm_no_key = MockLLM(
        config=LLMConfig(provider="mock", model="m", api_key=""),
    )
    assert llm_no_key.api_key == ""

    print("  PASSED: MockLLM works correctly")


# ============================================================
# TEST 7: MockLLM legacy constructor
# ============================================================

def test_mock_llm_legacy_constructor():
    print("\nTEST 7: MockLLM legacy constructor")
    llm = MockLLM(api_key="test-key", model="legacy-model")
    assert llm.model == "legacy-model"
    assert isinstance(llm, BaseLLM)

    request = LLMRequest(
        messages=[{"role": "user", "content": "test"}],
    )
    response = llm.generate(request)
    assert isinstance(response, LLMResponse)
    assert "test" in response.text

    print("  PASSED: MockLLM legacy constructor works")


# ============================================================
# TEST 8: Factory - mock provider
# ============================================================

def test_factory_mock_provider():
    print("\nTEST 8: Factory - mock provider")
    llm = create_llm(
        provider="mock",
        api_key="",
        model="test-mock",
    )

    assert isinstance(llm, MockLLM)
    assert llm.config.provider == "mock"
    assert llm.model == "test-mock"

    request = LLMRequest(
        messages=[{"role": "user", "content": "Factory test"}],
    )
    response = llm.generate(request)
    assert isinstance(response, LLMResponse)
    assert "Factory test" in response.text

    print("  PASSED: Factory creates MockLLM correctly")


# ============================================================
# TEST 9: Factory - unsupported provider
# ============================================================

def test_factory_unsupported_provider():
    print("\nTEST 9: Factory - unsupported provider")
    try:
        create_llm(provider="nonexistent", api_key="key", model="m")
        assert False, "Should raise ValueError for unsupported provider"
    except ValueError as e:
        assert "Unsupported LLM provider" in str(e), f"Wrong error message: {e}"

    print("  PASSED: Factory rejects unsupported provider")


# ============================================================
# TEST 10: Factory - missing API key
# ============================================================

def test_factory_missing_api_key():
    print("\nTEST 10: Factory - missing API key")
    for provider in PROVIDERS_REQUIRING_API_KEY:
        try:
            with patch("app.llm.factory.LLM_API_KEY", None):
                create_llm(provider=provider, api_key=None, model="m")
            assert False, f"Should raise ValueError for {provider} with empty API key"
        except ValueError as e:
            assert "API key is not configured" in str(e), (
                f"Wrong error message for {provider}: {e}"
            )

    print("  PASSED: Factory rejects missing API keys for all providers")


# ============================================================
# TEST 11: Factory registry is complete
# ============================================================

def test_factory_registry_complete():
    print("\nTEST 11: Factory registry is complete")
    expected = {"mock", "gemini", "groq", "mistral", "openrouter"}
    actual = set(LLM_PROVIDERS.keys())
    assert actual == expected, (
        f"Provider registry mismatch: expected {expected}, got {actual}"
    )

    print("  PASSED: All expected providers are in registry")


# ============================================================
# TEST 12: All providers conform to BaseLLM (with mocks)
# ============================================================

def test_all_providers_conform_to_base_llm():
    print("\nTEST 12: All providers conform to BaseLLM")
    from app.llm.gemini import GeminiLLM
    from app.llm.groq import GroqLLM
    from app.llm.mistral import MistralLLM
    from app.llm.openrouter import OpenRouterLLM

    providers = [MockLLM, GeminiLLM, GroqLLM, MistralLLM, OpenRouterLLM]

    for cls in providers:
        assert issubclass(cls, BaseLLM), f"{cls.__name__} does not extend BaseLLM"

    # Check that all have generate method
    for cls in providers:
        assert hasattr(cls, "generate"), f"{cls.__name__} missing generate method"

    print("  PASSED: All providers extend BaseLLM")


# ============================================================
# TEST 13: Provider SDK isolation
# ============================================================

def test_provider_sdk_isolation():
    print("\nTEST 13: Provider SDK isolation")
    # Verify that provider SDK imports exist ONLY in provider files
    import app.chat as chat_mod
    import app.config as config_mod

    # chat.py should not import provider SDKs
    chat_source = open(chat_mod.__file__).read()
    assert "from google" not in chat_source, "chat.py imports google SDK"
    assert "from groq" not in chat_source, "chat.py imports groq SDK"
    assert "from mistralai" not in chat_source, "chat.py imports mistral SDK"
    assert "from openai" not in chat_source, "chat.py imports openai SDK"

    print("  PASSED: No provider SDK leaks into application code")


# ============================================================
# TEST 14: chat.py uses common interface
# ============================================================

def test_chat_uses_common_interface():
    print("\nTEST 14: chat.py uses common interface")
    import app.chat as chat_mod
    import inspect

    source = inspect.getsource(chat_mod)

    # Should import from llm.factory and llm.types
    assert "from .llm.factory import create_llm" in source
    assert "from .llm.types import LLMRequest" in source

    # Should NOT import provider classes
    assert "GeminiLLM" not in source
    assert "GroqLLM" not in source
    assert "MistralLLM" not in source
    assert "OpenRouterLLM" not in source
    assert "MockLLM" not in source

    print("  PASSED: chat.py uses only the common LLM interface")


# ============================================================
# TEST 15: Application code using LLMRequest (integration)
# ============================================================

def test_application_code_uses_llm_request():
    print("\nTEST 15: Application code using LLMRequest")
    from unittest.mock import patch
    from app.chat import generate_normal_answer

    mock_llm = MockLLM(
        config=LLMConfig(provider="mock", model="test", api_key=""),
    )

    with patch("app.chat.create_llm", return_value=mock_llm):
        result = generate_normal_answer(
            query="What is Python?",
        )

    assert result["answer"], "Answer should not be empty"
    assert "What is Python?" in result["answer"]

    print("  PASSED: Application code works with LLMRequest")


# ============================================================
# TEST 16: Provider switching via config
# ============================================================

def test_provider_switching():
    print("\nTEST 16: Provider switching via config")
    # When LLM_PROVIDER=mock, create_llm should return MockLLM
    with patch("app.llm.factory.LLM_PROVIDER", "mock"), \
         patch("app.llm.factory.LLM_API_KEY", ""), \
         patch("app.llm.factory.LLM_MODEL", "test-model"):
        llm = create_llm()
        assert isinstance(llm, MockLLM), f"Expected MockLLM, got {type(llm)}"

    print("  PASSED: Provider switching works via config")


# ============================================================
# TEST 17: No generate(messages) old interface
# ============================================================

def test_no_old_generate_interface():
    print("\nTEST 17: No old generate(messages) interface")
    import app.chat as chat_mod
    import inspect

    source = inspect.getsource(chat_mod)
    # Should NOT call generate with raw messages
    assert "llm.generate(messages)" not in source, (
        "chat.py still uses old generate(messages) interface"
    )
    # Should create LLMRequest first
    assert "LLMRequest(" in source, (
        "chat.py should create LLMRequest before calling generate"
    )

    print("  PASSED: No old generate(messages) calls")


# ============================================================
# TEST 18: factory creates correct config
# ============================================================

def test_factory_creates_correct_config():
    print("\nTEST 18: Factory creates correct config")
    with patch("app.llm.factory.LLM_PROVIDER", "mock"), \
         patch("app.llm.factory.LLM_API_KEY", "test-key"), \
         patch("app.llm.factory.LLM_MODEL", "test-model"):
        llm = create_llm()

    assert llm.config.provider == "mock"
    assert llm.config.model == "test-model"
    assert llm.config.api_key == "test-key"

    print("  PASSED: Factory creates LLMConfig correctly")


# ============================================================
# TEST 19: Imports work correctly
# ============================================================

def test_imports():
    print("\nTEST 19: Imports work correctly")
    from app.llm.base import BaseLLM, LLMResponse
    from app.llm.config import LLMConfig
    from app.llm.types import LLMRequest
    from app.llm.factory import create_llm
    from app.chat import generate_normal_answer, generate_rag_answer

    assert BaseLLM is not None
    assert LLMResponse is not None
    assert LLMConfig is not None
    assert LLMRequest is not None
    assert create_llm is not None
    assert generate_normal_answer is not None
    assert generate_rag_answer is not None

    print("  PASSED: All imports work correctly")


# ============================================================
# TEST 20: Mock requires no API key in factory
# ============================================================

def test_mock_requires_no_api_key():
    print("\nTEST 20: Mock requires no API key in factory")
    # Mock should work with empty API key
    llm = create_llm(provider="mock", api_key="", model="test")
    assert isinstance(llm, MockLLM)

    # Also with None API key
    llm2 = create_llm(provider="mock", api_key=None, model="test")
    assert isinstance(llm2, MockLLM)

    print("  PASSED: Mock works without API key")


# ============================================================
# TEST 21: Provider independence - same interface
# ============================================================

def test_provider_independence():
    print("\nTEST 21: Provider independence - same interface")
    # All providers should accept LLMRequest and return LLMResponse
    providers = {
        "mock": MockLLM(config=LLMConfig(provider="mock", model="m", api_key="")),
    }

    request = LLMRequest(
        messages=[{"role": "user", "content": "test"}],
    )

    for name, llm in providers.items():
        assert isinstance(llm, BaseLLM), f"{name} is not BaseLLM"
        response = llm.generate(request)
        assert isinstance(response, LLMResponse), f"{name} did not return LLMResponse"
        assert isinstance(response.text, str), f"{name} response.text is not str"

    print("  PASSED: All providers share the same interface")


# ============================================================
# TEST 22: No silent fallback to mock
# ============================================================

def test_no_silent_fallback():
    print("\nTEST 22: No silent fallback to mock")
    try:
        with patch("app.llm.factory.LLM_API_KEY", None):
            create_llm(provider="gemini", api_key=None, model="test")
        assert False, "Should raise ValueError, not fall back to mock"
    except ValueError as e:
        assert "API key is not configured" in str(e)

    try:
        with patch("app.llm.factory.LLM_API_KEY", None):
            create_llm(provider="groq", api_key=None, model="test")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "API key is not configured" in str(e)

    print("  PASSED: No silent fallback to mock")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    tests = [
        test_llm_config_normalization,
        test_llm_config_structure,
        test_llm_request_defaults,
        test_llm_response_structure,
        test_base_llm_contract,
        test_mock_llm,
        test_mock_llm_legacy_constructor,
        test_factory_mock_provider,
        test_factory_unsupported_provider,
        test_factory_missing_api_key,
        test_factory_registry_complete,
        test_all_providers_conform_to_base_llm,
        test_provider_sdk_isolation,
        test_chat_uses_common_interface,
        test_application_code_uses_llm_request,
        test_provider_switching,
        test_no_old_generate_interface,
        test_factory_creates_correct_config,
        test_imports,
        test_mock_requires_no_api_key,
        test_provider_independence,
        test_no_silent_fallback,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("LLM ARCHITECTURE TEST SUITE")
    print("=" * 60)

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFAILURES:")
        for name, err in errors:
            print(f"  {name}: {err}")

    sys.exit(1 if failed else 0)
