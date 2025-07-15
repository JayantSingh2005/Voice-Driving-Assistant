from together import Together
from dotenv import load_dotenv
import os
import httpx

load_dotenv()
# NOTE: Never hardcode API keys in production. Use environment variables.
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

client = Together()  # Uses TOGETHER_API_KEY from env

def normalize_text(text: str) -> str:
    """
    Calls Together API (Gemma-2 or similar) to normalize Hinglish text.
    """
    prompt = (
        "You are a Hinglish text normalizer. "
        "Correct spelling, grammar, and phonetic errors in the following text, but do not translate to pure Hindi or English. "
        "Keep code-mixed style. Only return the corrected text.\n\n"
        f"Input: {text}\nOutput:"
    )
    url = "https://api.together.xyz/v1/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gemma-2b-it",  # or "gemma-7b-it" or "gemma-27b-it"
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.2,
        "stop": ["\n"]
    }
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[Normalizer] Error: {e}")
        return text  # Fallback: return original

def normalize_transcript_with_gemma(transcript: str) -> str:
    prompt = f"""
You are a Hinglish transcription corrector.
Fix spelling, grammar, and place names in spoken text from speech-to-text errors.
Preserve meaning and named locations like India Gate, CP, Sarojini.

Examples:
- Input: indiya get le chao
  Output: India Gate le chalo

- Input: sarojni nagar chalo
  Output: Sarojini Nagar chalo

Now fix this one:
Input: {transcript}
Output:
"""
    response = client.chat.completions.create(
        model="google/gemma-2-27b-it",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    text = response.choices[0].message.content
    if "Output:" in text:
        return text.split("Output:")[-1].strip()
    return text.strip()

# For direct testing
if __name__ == "__main__":
    test_cases = [
        "indiya get le chao",
        "sarojni nagar chalo",
        "cp le chao",
        "qutub minar le chao",
        "traphic kaisa hai",
        "bhai indiya get tk ka route set kr de",
    ]
    for text in test_cases:
        print(f"\nInput: {text}")
        try:
            normalized = normalize_text(text)
            print(f"Normalized: {normalized}")
        except Exception as e:
            print(f"Error: {e}") 