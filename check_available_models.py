#!/usr/bin/env python3
"""Check which OpenAI models are available in your API account."""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in .env file")
    sys.exit(1)

client = OpenAI(api_key=api_key)

print("Checking available OpenAI models...")
print("=" * 60)

# List of models to check
models_to_check = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.2-instant",
    "gpt-5.2-thinking",
    "gpt-5.2-pro",
]

available_models = []
unavailable_models = []

for model in models_to_check:
    try:
        # Try a simple completion to test if model is available
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        available_models.append(model)
        print(f"✓ {model}: Available")
    except Exception as e:
        unavailable_models.append(model)
        error_msg = str(e)
        if "model_not_found" in error_msg.lower() or "does not exist" in error_msg.lower():
            print(f"✗ {model}: Not found")
        elif "rate limit" in error_msg.lower():
            print(f"? {model}: Rate limited (may be available)")
        else:
            print(f"✗ {model}: Error - {error_msg[:50]}")

print("\n" + "=" * 60)
print(f"Available models: {len(available_models)}")
print(f"Unavailable models: {len(unavailable_models)}")

if available_models:
    print("\nRecommended config.yaml models list:")
    print("models:")
    for model in available_models:
        print(f"  - \"{model}\"")

