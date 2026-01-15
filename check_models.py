#!/usr/bin/env python3
"""Check available models for each provider."""
import os
from dotenv import load_dotenv

load_dotenv()

print("Checking available models for each provider...")
print("=" * 60)

# Check OpenAI
print("\nOpenAI Models:")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    models = client.models.list()
    openai_models = [m.id for m in models.data if 'gpt' in m.id.lower()]
    for model in sorted(openai_models)[:10]:  # Show first 10
        print(f"  - {model}")
except Exception as e:
    print(f"  Error: {e}")

# Check Gemini
print("\nGemini Models:")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    models = genai.list_models()
    print("  Available models:")
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            # Extract just the model name (remove 'models/' prefix)
            model_name = model.name.replace('models/', '')
            print(f"  - {model_name} (full: {model.name})")
except Exception as e:
    print(f"  Error: {e}")
    print("  Try these common model names:")
    print("    - gemini-1.5-pro-latest")
    print("    - gemini-1.5-flash-latest")
    print("    - gemini-pro")

# Check Anthropic
print("\nAnthropic Models:")
print("  Common model names to try:")
print("  - claude-3-5-sonnet-20240620")
print("  - claude-3-5-haiku-20241022")
print("  - claude-3-opus-20240229")
print("  - claude-3-sonnet-20240229")
print("  - claude-3-haiku-20240307")
print("  - claude-3-5-sonnet")
print("  - claude-3-5-haiku")
try:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # Anthropic doesn't have a list_models endpoint
    # Test a simple model to verify API key works
    print("  (Anthropic doesn't provide a model list endpoint)")
    print("  Try the model names above - the API will error if model doesn't exist")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)

