import ast
import os
import re
from typing import Any, Dict, Optional

from Extropic_Integration.cockpit.llm_inference import invoke_llm_for_raw_json

# --- Configuration ---
EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../experiments"))

# --- System Prompts ---
ARCHITECT_SYSTEM_PROMPT = """
You are the "Architect", an expert Python developer specializing in the D-ND Omega Kernel.
Your goal is to generate Python experiments based on user prompts.

The D-ND Omega Kernel (SACS) has the following key components:
- `CognitiveField`: Represents the dimensional space.
- `OmegaKernel`: The core processing unit.
- `MetricTensor`: Defines the geometry of the field.

Your output MUST be a JSON object with the following structure:
{
    "code": "The complete, executable Python code for the experiment.",
    "explanation": "A brief explanation of what the code does and the theoretical concepts involved.",
    "filename": "suggested_filename.py"
}

The generated code MUST:
1. Import necessary modules (assume `dnd_kernel` package structure).
2. Define a function `run_experiment()` that executes the logic.
3. Be syntactically correct and follow PEP 8.
4. Include comments explaining key steps.

Example Import:
from dnd_kernel.genesis import CognitiveField
from dnd_kernel.omega import OmegaKernel

If the user asks for a specific concept (e.g., "Fibonacci"), ensure the logic reflects it.
"""


async def generate_experiment_code(prompt: str, model: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates Python experiment code using the LLM.
    """
    user_prompt = f"Create a D-ND Omega Kernel experiment for: {prompt}"

    try:
        response = await invoke_llm_for_raw_json(
            model_name=model, system_prompt=ARCHITECT_SYSTEM_PROMPT, user_prompt=user_prompt, api_key=api_key
        )

        # Basic validation of the generated code
        code = response.get("code", "")
        validation = validate_python_code(code)

        if not validation["valid"]:
            response["warning"] = f"Generated code has syntax errors: {validation['error']}"

        return response

    except Exception as e:
        print(f"[Forge] Generation Error: {e}")
        return {"code": "", "explanation": f"Failed to generate experiment: {str(e)}", "filename": "error.py"}


def validate_python_code(code: str) -> Dict[str, Any]:
    """
    Validates Python code syntax using the AST module.
    """
    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": str(e)}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def save_experiment_file(code: str, filename: str) -> Dict[str, Any]:
    """
    Saves the generated code to the experiments directory.
    """
    try:
        if not os.path.exists(EXPERIMENTS_DIR):
            os.makedirs(EXPERIMENTS_DIR)

        # Sanitize filename
        filename = re.sub(r"[^\w\-_.]", "", filename)
        if not filename.endswith(".py"):
            filename += ".py"

        file_path = os.path.join(EXPERIMENTS_DIR, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        return {"success": True, "path": file_path, "filename": filename}

    except Exception as e:
        return {"success": False, "error": str(e)}


# --- Widget Forge Logic ---

WIDGET_SYSTEM_PROMPT = """
You are the "Frontend Architect", an expert React/Tailwind developer for the SACS Cockpit.
Your goal is to generate JSON configuration for dashboard widgets.

The user will describe a desired visualization (e.g., "A purple bar chart showing revenue vs costs").
You must output a VALID JSON object adhering to this `WidgetConfig` schema:

{
  "config": {
    "id": "generated_id",
    "title": "Widget Title",
    "type": "bar" | "line" | "area" | "pie" | "radar" | "radial",
    "dataSource": "management" | "cashflow" | "rating" | "hr",
    "dataKeys": ["key1", "key2"],
    "colorTheme": "cyan" | "purple" | "green" | "orange",
    "isSystem": false,
    "isVisible": true,
    "colSpan": 1 or 2
  },
  "explanation": "Brief explanation of design choices."
}

Rules:
- `dataSource`: Choose purely based on context (Revenue/Profit -> 'management', Cash -> 'cashflow').
- `dataKeys`: Must be valid keys for the source (e.g. 'revenue', 'costs', 'profit' for management).
- `colorTheme`: Use 'cyan' for logic, 'purple' for physics/entropy, 'green' for profit, 'orange' for risk.
"""


async def generate_widget_config(prompt: str, model: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates React Widget Config JSON using the LLM.
    """
    try:
        response = await invoke_llm_for_raw_json(
            model_name=model,
            system_prompt=WIDGET_SYSTEM_PROMPT,
            user_prompt=f"Create a widget for: {prompt}",
            api_key=api_key,
        )
        # Ensure a unique ID timestamp to force React refresh if needed
        if "config" in response:
            import time

            response["config"]["id"] = f"ai_gen_{int(time.time())}"

        return response

    except Exception as e:
        print(f"[Widget Forge] Generation Error: {e}")
        return {
            "config": {
                "id": "error_fallback",
                "title": "Generation Failed",
                "type": "bar",
                "dataSource": "management",
                "dataKeys": ["revenue"],
                "colorTheme": "orange",
                "isSystem": False,
                "isVisible": True,
                "colSpan": 1,
            },
            "explanation": f"LLM Error: {str(e)}",
        }
