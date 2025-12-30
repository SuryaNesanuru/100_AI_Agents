
import json
import re
from datetime import date
from openai import OpenAI
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
client = OpenAI()  # expects OPENAI_API_KEY in env

SYSTEM_PROMPT = """
You are an Email Summarization Agent.

Your job:
1. Summarize the email in 2–3 sentences
2. Extract key points
3. Extract action items (who should do what)
4. Identify deadlines
5. Classify urgency: Low, Medium, or High

Return ONLY valid JSON with this schema:

{
  "summary": "",
  "key_points": [],
  "action_items": [],
  "deadlines": [],
  "urgency": ""
}
"""

def read_email(path="email.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _extract_content_from_choice(choice):
    # Support various SDK shapes: choice.message.content, choice['message']['content'], choice.text, etc.
    if not choice:
        return ""
    # try object-style access
    msg = getattr(choice, "message", None)
    if msg:
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
        return getattr(msg, "content", "") or str(msg)
    # try dict-like or legacy
    try:
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                return choice["message"].get("content", "") or ""
            return choice.get("text", "") or ""
    except Exception:
        pass
    # fallback to attributes
    return getattr(choice, "text", "") or str(choice)

def _parse_json_from_text(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response text")
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract a JSON object embedded in text
        m = re.search(r"(\{.*\})", text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    raise

def summarize_email(email_text):
    response = client.chat.completions.create(
        # use a broadly available chat model; change as needed
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text}
        ],
        temperature=0.2
    )

    # extract text safely from first choice
    choice = None
    try:
        choice = response.choices[0]
    except Exception:
        # some SDKs return top-level 'choice' or different shape
        try:
            choice = (response["choices"][0]) if isinstance(response, dict) else None
        except Exception:
            choice = None

    content_text = _extract_content_from_choice(choice)

    # try strict JSON, then best-effort extraction, then fallback structure
    try:
        return _parse_json_from_text(content_text)
    except Exception:
        # fallback: return a best-effort structure using the raw content as the summary
        return {
            "summary": content_text.strip(),
            "key_points": [],
            "action_items": [],
            "deadlines": [],
            "urgency": "Unknown"
        }

def save_outputs(data):
    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Email Summary ({date.today()})\n")
        f.write("=" * 40 + "\n\n")
        f.write("SUMMARY:\n")
        f.write(data.get("summary", "") + "\n\n")

        f.write("KEY POINTS:\n")
        for p in data.get("key_points", []):
            f.write(f"- {p}\n")

        f.write("\nACTION ITEMS:\n")
        for a in data.get("action_items", []):
            f.write(f"- {a}\n")

        f.write("\nDEADLINES:\n")
        for d in data.get("deadlines", []):
            f.write(f"- {d}\n")

        f.write(f"\nURGENCY: {data.get('urgency', 'Unknown')}\n")

def main():
    email_text = read_email()
    result = summarize_email(email_text)
    save_outputs(result)
    print("Email summarized successfully.")
    print(result)

if __name__ == "__main__":
    main()