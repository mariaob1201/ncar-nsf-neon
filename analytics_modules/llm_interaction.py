import os
from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Set it in your environment or analytics_modules/.env to use ask_llm()."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def ask_llm(user_prompt, context=None, temperature=0.2):
    """
    Send a user prompt to the LLM with optional scientific context.
    """
    messages = []

    if context:
        messages.append({
            "role": "system",
            "content": context
        })

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    response = _get_client().chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content