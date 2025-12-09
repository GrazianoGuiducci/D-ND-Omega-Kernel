import json
import os
from typing import Any, AsyncGenerator, Dict, Optional

from dotenv import load_dotenv
from fastapi import HTTPException, Request
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SYSTEM_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _resolve_api_key(request_key: Optional[str] = None) -> str:
    """
    Resolves the API key to use with the following precedence:
    1. User-provided key (BYOK) via header
    2. System-configured key (Env)
    """
    if request_key and request_key.strip():
        return request_key.strip()

    if SYSTEM_API_KEY and SYSTEM_API_KEY.strip():
        return SYSTEM_API_KEY.strip()

    raise HTTPException(status_code=401, detail="OPENROUTER_API_KEY_MISSING")


def _get_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def _map_openrouter_error(e: Exception) -> None:
    """Maps OpenRouter errors to specific HTTP exceptions."""
    text = str(e).lower()
    if "insufficient funds" in text or "payment required" in text:
        raise HTTPException(status_code=503, detail="OPENROUTER_SYSTEM_KEY_EXHAUSTED")
    # Re-raise original exception if not mapped
    raise e


async def invoke_llm_for_raw_json(
    model_name: str, system_prompt: str, user_prompt: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Invokes LLM and expects a JSON response.
    """
    try:
        resolved_key = _resolve_api_key(api_key)
        client = _get_client(resolved_key)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content:
            return json.loads(content)
        return {}

    except Exception as e:
        _map_openrouter_error(e)
        print(f"[invoke_llm_for_raw_json] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_llm_response(
    model_name: str, system_prompt: str, user_prompt: str, api_key: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streams LLM response yielding chunks and usage data.
    """
    try:
        resolved_key = _resolve_api_key(api_key)
        client = _get_client(resolved_key)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},  # OpenRouter specific
        )

        for chunk in stream:
            # Handle Usage
            if chunk.usage:
                yield {
                    "event": "usage",
                    "data": {"usage": chunk.usage.model_dump(), "model": model_name, "provider": "OpenRouter"},
                }
                continue

            # Handle Content
            if chunk.choices and chunk.choices[0].delta.content:
                yield {"event": "message_chunk", "data": {"chunk": chunk.choices[0].delta.content}}

        yield {"event": "final_response", "data": {"response": "DONE"}}

    except Exception as e:
        try:
            _map_openrouter_error(e)
        except HTTPException as he:
            yield {"event": "error", "data": {"code": he.detail, "message": "System Key Exhausted or API Error"}}
            return

        print(f"[stream_llm_response] ERROR: {e}")
        yield {"event": "error", "data": {"error": str(e)}}


def get_openrouter_status(request: Request) -> Dict[str, Any]:
    """
    Returns the status of API keys (System vs User).
    """
    user_key = request.headers.get("X-OpenRouter-Key")
    has_user_key = bool(user_key and user_key.strip())
    has_server_key = bool(SYSTEM_API_KEY and SYSTEM_API_KEY.strip())

    key_source = "none"
    if has_user_key:
        key_source = "user"
    elif has_server_key:
        key_source = "server"

    return {"key_source": key_source, "has_server_key": has_server_key, "has_user_key": has_user_key}


async def get_available_models(api_key: Optional[str] = None) -> Any:
    """
    Fetches available models from OpenRouter.
    """
    try:
        resolved_key = _resolve_api_key(api_key)
        # client = _get_client(resolved_key) # Unused for direct http call if needed, but if we use requests:

        # OpenRouter specific endpoint for models
        # We use the requests library or standard http client as openai lib might not have a direct 'models.list' that maps perfectly to OR's extra fields if we want them,
        # but client.models.list() should work for basic info.
        # For full info (pricing etc), we might need a direct call.
        # Let's try client.models.list() first, but OpenRouter returns a specific structure.
        # Actually, standard OpenAI client.models.list() returns Model objects.
        # OpenRouter's /models endpoint returns a list of models with pricing info.

        import httpx

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f"{OPENROUTER_BASE_URL}/models", headers={"Authorization": f"Bearer {resolved_key}"}
            )
            response.raise_for_status()
            data = response.json()
            return data["data"]  # OpenRouter returns { data: [...] }

    except Exception as e:
        print(f"[get_available_models] ERROR: {e}")
        # Fallback if request fails
        return []
