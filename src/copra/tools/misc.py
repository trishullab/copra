def is_open_ai_model(model_name: str) -> bool:
    """
    Check if the model name is an OpenAI model.
    """
    return model_name.startswith("gpt-") or \
    model_name.startswith("gpt-4-") or \
    model_name.startswith("gpt-3.5-") or \
    model_name.startswith("o1") or \
    model_name.startswith("o1") or \
    model_name.startswith("o3") or \
    model_name.startswith("o3-") or \
    model_name.startswith("o") or \
    model_name.startswith("o-")

def is_anthropic_model(model_name: str) -> bool:
    """
    Check if the model name is an Anthropic model.
    """
    return model_name.startswith("claude") or \
    model_name.startswith("claude-") or \
    model_name.startswith("claude-1") or \
    model_name.startswith("claude-2") or \
    model_name.startswith("claude-3") or \
    model_name.startswith("claude-4")

def is_bedrock_model(model_name: str) -> bool:
    """
    Check if the model name is a Bedrock model.
    """
    return model_name.startswith("anthropic.") or \
    model_name.startswith("deepseek.")

def is_vllm_model(model_name: str) -> bool:
    """
    Check if the model name is a vLLM model.
    vLLM models use the prefix "vllm:" followed by the HuggingFace model name.
    Example: vllm:AI-MO/Kimina-Prover-Distill-1.7B
    """
    return model_name.startswith("vllm:")

def model_supports_openai_api(model_name: str) -> bool:
    """
    Check if the model supports OpenAI API.
    """
    return is_open_ai_model(model_name) or \
    is_anthropic_model(model_name) or \
    is_bedrock_model(model_name) or \
    is_vllm_model(model_name)
