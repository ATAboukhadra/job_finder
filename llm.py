"""Local LLM integration via Ollama.

Model recommendations by hardware:
  - NVIDIA GPU (CUDA) or Apple Silicon (MPS): qwen3.5:9b  (~5 GB)
  - CPU only:                                 qwen2.5:3b  (~2 GB, runs on 8 GB RAM)
"""

import json
import logging
import os
import platform
import subprocess
import time
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
CPU_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT_SECONDS = 900  # CPU can be slow; allow long generations by default.


def _default_timeout() -> int:
    """Configurable default timeout for Ollama requests."""
    raw = os.environ.get("OLLAMA_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except Exception:
            pass
    return DEFAULT_TIMEOUT_SECONDS


def detect_hardware() -> str:
    """Return 'cuda', 'mps', or 'cpu'."""
    # Try torch first (most reliable)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        pass
    # Fallback: nvidia-smi
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return "cuda"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Apple Silicon
    if platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin":
        return "mps"
    return "cpu"


def recommend_model() -> str:
    """Return the best Ollama model for the current hardware."""
    hw = detect_hardware()
    if hw in ("cuda", "mps"):
        logger.info(f"Hardware detected: {hw} — using {DEFAULT_MODEL}")
        return DEFAULT_MODEL
    else:
        logger.info(f"Hardware detected: cpu — using lighter model {CPU_MODEL}")
        return CPU_MODEL


def check_ollama_available() -> bool:
    """Check if Ollama server is running and responsive."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def list_models() -> list:
    """List available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def generate(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Generate text using Ollama.

    Args:
        prompt: The user prompt.
        system: System prompt for context/instructions.
        model: Ollama model name.
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        Generated text string.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    logger.info("LLM generate: model=%s, prompt_len=%d", model, len(prompt))
    start = time.time()

    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=timeout or _default_timeout(),
        )
        r.raise_for_status()
        result = r.json()
        text = result.get("response", "")
        elapsed = time.time() - start
        logger.info("LLM response: %d chars in %.1fs", len(text), elapsed)
        return text.strip()
    except requests.Timeout:
        logger.error("LLM request timed out after %ds", timeout)
        raise
    except requests.RequestException as e:
        logger.error("LLM request failed: %s", e)
        raise


def generate_structured(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Generate structured JSON output from LLM.

    The prompt should instruct the model to return valid JSON.
    Attempts to parse the response as JSON, with retry on failure.
    """
    json_system = system + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, no code fences."

    for attempt in range(3):
        text = generate(
            prompt=prompt,
            system=json_system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout or _default_timeout(),
        )

        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, e)
            if attempt < 2:
                prompt = (
                    f"Your previous response was not valid JSON. Error: {e}\n"
                    f"Please try again. Return ONLY valid JSON.\n\n"
                    f"Original request:\n{prompt}"
                )

    logger.error("Failed to get valid JSON after 3 attempts")
    return {}


def generate_latex(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Generate LaTeX content, stripping any markdown fences."""
    text = generate(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout or _default_timeout(),
    )

    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```latex or ```) and last line (```)
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    return cleaned.strip()
