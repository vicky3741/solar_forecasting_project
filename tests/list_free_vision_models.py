"""
=========================================================
Solar Forecasting Project
List Free Vision Models on OpenRouter (live)
=========================================================
Queries OpenRouter's public model catalogue and prints the
models that are BOTH free AND accept image input - i.e. the
current, real alternatives to Gemini for reading Windy cloud
frames. The catalogue changes constantly, so this always
shows what is actually available right now (no API key
needed - the models endpoint is public).

Any id printed here can be dropped straight into
settings.yaml as vision.openai_model (with
openai_base_url: https://openrouter.ai/api/v1) and tested
with tests/test_llm_comparison.py.

Run:  python -m tests.list_free_vision_models
=========================================================
"""

import requests

MODELS_URL = "https://openrouter.ai/api/v1/models"


def is_free(model):
    pricing = model.get("pricing", {}) or {}
    try:
        prompt = float(pricing.get("prompt", 0) or 0)
        completion = float(pricing.get("completion", 0) or 0)
    except (TypeError, ValueError):
        prompt = completion = 1
    return (prompt == 0 and completion == 0) or model.get("id", "").endswith(":free")


def supports_image(model):
    arch = model.get("architecture", {}) or {}
    modalities = arch.get("input_modalities") or arch.get("modality", "")
    if isinstance(modalities, str):
        return "image" in modalities
    return "image" in modalities


def main():

    print("Fetching live model list from OpenRouter ...")
    data = requests.get(MODELS_URL, timeout=30).json().get("data", [])

    free_vision = [
        m for m in data if is_free(m) and supports_image(m)
    ]

    free_vision.sort(key=lambda m: m.get("context_length", 0), reverse=True)

    print(f"\n{len(free_vision)} FREE vision models available right now:\n")
    print(f"  {'model id':52s} {'context':>9s}")
    print("  " + "-" * 63)
    for m in free_vision:
        print(f"  {m['id']:52s} {str(m.get('context_length', '?')):>9s}")

    print("\nTo test one:")
    print("  1. set  vision.openai_base_url: https://openrouter.ai/api/v1")
    print("  2. set  vision.openai_model: <one of the ids above>")
    print("  3. put  OPENROUTER_API_KEY=... in .env  (free signup, no card)")
    print("  4. run  python -m tests.test_llm_comparison")


if __name__ == "__main__":
    main()
