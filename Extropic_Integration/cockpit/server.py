"""
SACS Cockpit Server
Exposes the SACS Kernel via FastAPI for real-time visualization and interaction.
"""

import json
import os
import sys
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.responses import StreamingResponse

from Extropic_Integration.architect.sacs import SACS
from Extropic_Integration.cockpit.forge_service import (
    generate_experiment_code,
    generate_widget_config,
    save_experiment_file,
)
from Extropic_Integration.cockpit.llm_inference import (
    get_available_models,
    get_openrouter_status,
    invoke_llm_for_raw_json,
    stream_llm_response,
)

app = FastAPI(title="SACS Cockpit", version="1.0")


# Initialize SACS
# We use a smaller size for the web demo to ensure responsiveness
sacs = SACS(size=50)


# --- Data Models ---
class IntentRequest(BaseModel):
    intent: str
    steps: int = 300


class CycleResponse(BaseModel):
    manifesto: str
    metrics: Dict[str, float]
    dipoles: List[Any]
    taxonomy_update: bool
    didactic: Dict[str, Any] = {}


# --- Helpers ---
def to_serializable(obj):
    """Recursively converts JAX/Numpy arrays to Python native types."""
    if hasattr(obj, "tolist"):  # JAX/Numpy arrays
        return obj.tolist()
    if hasattr(obj, "item"):  # Scalar arrays
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


# --- Endpoints ---


@app.post("/api/intent", response_model=CycleResponse)
async def process_intent(request: IntentRequest):
    """
    Injects an intent into the SACS Kernel and runs a full cycle.
    """
    print(f"[Cockpit] Processing intent: {request.intent}")
    try:
        # Run the cycle
        # sacs.process now returns {"manifesto": str, "result": dict}
        sacs_output = sacs.process(request.intent, steps=request.steps)
        manifesto = sacs_output["manifesto"]
        result = sacs_output["result"]

        # Gather metrics from the last run
        # Gather metrics from the last run
        metrics = {
            "coherence": to_serializable(result["coherence"]),
            "tension": to_serializable(result["tension"]),
            "logic_density": to_serializable(sacs.omega.logic_density),
            "energy": to_serializable(result["energy"]),
        }

        # Get dipoles from Sonar
        dipoles = []
        if sacs.archivista.memory["cycles"]:
            last_entry = sacs.archivista.memory["cycles"][-1]
            dipoles = last_entry.get("dipoles", [])

        # Prepare didactic info
        didactic_info = to_serializable(result.get("didactic", {}))

        return CycleResponse(
            manifesto=manifesto, metrics=metrics, dipoles=dipoles, taxonomy_update=True, didactic=didactic_info
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/state")
async def get_state():
    """
    Returns the current internal state for visualization.
    """
    # We want to visualize the MetricTensor (Telaio) and the Spin State (Omega)
    # MetricTensor is in sacs.omega.metric_tensor (if we modified omega to store it)
    # Actually, omega.crystallize uses it.
    # Let's return the logic_density and current energy.

    return {
        "logic_density": float(sacs.omega.logic_density),
        "experience": sacs.omega.experience,
        "memory_size": len(sacs.archivista.memory["cycles"]),
        "taxonomy": sacs.archivista.memory["taxonomy"],
    }


@app.post("/api/reset")
async def reset_memory():
    """
    Resets the system memory and re-initializes SACS components.
    Used for the Genesis Protocol.
    """
    print("[Cockpit] Resetting System Memory...")
    try:
        # Clear memory file
        if os.path.exists("system_memory.json"):
            os.remove("system_memory.json")

        # Re-init Archivista to reload empty memory
        sacs.archivista.memory = {"cycles": [], "taxonomy": {}}

        # Reset Omega state (optional, but good for clean slate)
        sacs.omega.logic_density = 0.2
        sacs.omega.experience = 0

        return {"status": "System Reset Complete", "memory_size": 0}
    except Exception as e:
        print(f"Error resetting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/docs")
async def list_docs():
    """Lists available documentation files."""
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs"))
    files = []
    if os.path.exists(docs_dir):
        for f in os.listdir(docs_dir):
            if f.endswith(".md"):
                files.append(f)
    return {"files": files}


@app.get("/api/docs/{filename}")
async def get_doc(filename: str):
    """Retrieves the content of a documentation file."""
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs"))
    file_path = os.path.join(docs_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {"content": content}


# ... (previous imports)


# ... (previous code)

# --- OpenRouter Endpoints ---


@app.get("/openrouter/status")
async def openrouter_status_endpoint(request: Request):
    """
    Returns the status of OpenRouter keys (System vs User).
    """
    return get_openrouter_status(request)


@app.get("/api/v1/openrouter/models")
async def openrouter_models_endpoint(request: Request):
    """
    Returns available models from OpenRouter.
    """
    user_key = request.headers.get("X-OpenRouter-Key")
    return await get_available_models(user_key)


class LlmInvokeRequest(BaseModel):
    model: str
    system_prompt: str
    user_prompt: str


@app.post("/api/llm/invoke")
async def invoke_llm_endpoint(request: Request, payload: LlmInvokeRequest):
    """
    Invokes LLM for JSON response.
    """
    user_key = request.headers.get("X-OpenRouter-Key")
    return await invoke_llm_for_raw_json(
        model_name=payload.model, system_prompt=payload.system_prompt, user_prompt=payload.user_prompt, api_key=user_key
    )


@app.post("/api/llm/stream")
async def stream_llm_endpoint(request: Request, payload: LlmInvokeRequest):
    """
    Streams LLM response via SSE.
    """
    user_key = request.headers.get("X-OpenRouter-Key")

    async def event_generator():
        async for event in stream_llm_response(
            model_name=payload.model,
            system_prompt=payload.system_prompt,
            user_prompt=payload.user_prompt,
            api_key=user_key,
        ):
            # SSE Format: data: <json>\n\n
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Forge Endpoints ---


class ForgeGenerateRequest(BaseModel):
    prompt: str
    model: str = "google/gemini-2.0-flash-exp:free"  # Default model


class ForgeInjectRequest(BaseModel):
    code: str
    filename: str


@app.post("/api/forge/generate")
async def forge_generate_endpoint(request: Request, payload: ForgeGenerateRequest):
    """
    Generates experiment code via LLM.
    """
    user_key = request.headers.get("X-OpenRouter-Key")
    return await generate_experiment_code(prompt=payload.prompt, model=payload.model, api_key=user_key)


@app.post("/api/forge/inject")
async def forge_inject_endpoint(payload: ForgeInjectRequest):
    """
    Saves the generated experiment code to the server.
    """
    result = save_experiment_file(payload.code, payload.filename)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


# --- Widget Forge ---


class WidgetGenerateRequest(BaseModel):
    prompt: str
    model: str = "google/gemini-2.0-flash-exp:free"


@app.post("/api/forge/widget")
async def forge_widget_endpoint(request: Request, payload: WidgetGenerateRequest):
    """
    Generates a React Widget Configuration JSON via LLM.
    """
    user_key = request.headers.get("X-OpenRouter-Key")
    return await generate_widget_config(prompt=payload.prompt, model=payload.model, api_key=user_key)


# --- Widget Persistence ---

WIDGETS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../custom_widgets.json"))


class SaveWidgetRequest(BaseModel):
    widgets: List[Dict[str, Any]]


@app.get("/api/widgets")
async def get_widgets():
    """
    Loads saved custom widgets.
    """
    if not os.path.exists(WIDGETS_FILE):
        return {"widgets": []}

    try:
        with open(WIDGETS_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                return {"widgets": []}
            return {"widgets": json.loads(content)}
    except Exception as e:
        print(f"[Widgets] Load Error: {e}")
        return {"widgets": []}


@app.post("/api/widgets")
async def save_widgets(payload: SaveWidgetRequest):
    """
    Saves the list of custom widgets.
    """
    try:
        with open(WIDGETS_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload.widgets, indent=2))
        return {"success": True}
    except Exception as e:
        print(f"[Widgets] Save Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ... (rest of server.py)


@app.delete("/api/widgets/{widget_id}")
async def delete_widget(widget_id: str):
    """
    Deletes a single widget by its ID.
    """
    try:
        # Load current widgets
        widgets = []
        if os.path.exists(WIDGETS_FILE):
            with open(WIDGETS_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    widgets = json.loads(content)

        # Find and remove the widget
        original_count = len(widgets)
        widgets = [w for w in widgets if w.get("id") != widget_id]

        if len(widgets) == original_count:
            raise HTTPException(status_code=404, detail=f"Widget {widget_id} not found")

        # Save updated list
        with open(WIDGETS_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(widgets, indent=2))

        return {"success": True, "deleted": widget_id}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Widgets] Delete Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
