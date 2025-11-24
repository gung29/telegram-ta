import os
import requests
import json

API_KEY = "AIzaSyB_AwxfLHALbBKCuwJz9xYfuxHXOWqdKro"

MODEL = "gemini-2.5-flash"  # bisa diganti sesuai kebutuhan
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
def classify_text(text: str) -> dict:
    """
    Kirim teks ke Gemini dan minta output JSON:
      {
        "score": number 0-100,
        "category": string,
        "reasoning": string
      }
    """
    prompt = f"""
    Analyze the following text for hate speech, toxicity, or harmful content.
    Text: "{text}"

    Return a JSON object with:
    - score: A number between 0 and 100 representing the likelihood of hate speech (100 being certain hate speech).
    - category: A short string classification (e.g., "Hate Speech", "Harassment", "Spam", "Safe", "Controversial").
    - reasoning: A brief explanation (max 1 sentence).
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "score": {"type": "NUMBER"},
                    "category": {"type": "STRING"},
                    "reasoning": {"type": "STRING"}
                },
                "required": ["score", "category", "reasoning"]
            }
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    resp = requests.post(ENDPOINT, headers=headers, json=payload)
    resp.raise_for_status()

    data = resp.json()
    # Struktur response Gemini: { "candidates": [ { "content": {...} } ] }
    # Karena kita minta responseMimeType application/json, biasanya hasilnya ada di candidates[0].content.parts[0].text
    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(raw_text)  # parse JSON string jadi dict
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw response:", json.dumps(data, indent=2))
        raise

    return result

if __name__ == "__main__":
    # Contoh: custom input dari user
    user_text = input("Masukkan teks yang ingin diklasifikasikan: ")
    result = classify_text(user_text)
    print("Hasil klasifikasi:")
    print(json.dumps(result, indent=2))
