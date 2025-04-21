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