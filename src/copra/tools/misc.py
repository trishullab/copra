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

def model_supports_openai_api(model_name: str) -> bool:
    """
    Check if the model supports OpenAI API.
    """
    return is_open_ai_model(model_name) or \
    is_anthropic_model(model_name)