import base64
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ---------- HELPERS ----------
def ensure_list(value):
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        return [s.strip() for s in value.split(".") if s.strip()]
    return []

def clean_result(result):
    result["handling_procedure"] = ensure_list(result.get("handling_procedure"))
    result["recyclability"] = ensure_list(result.get("recyclability"))
    result["environmental_impact"] = ensure_list(result.get("environmental_impact"))

    # Fallbacks
    if not result["handling_procedure"]:
        result["handling_procedure"] = [
            "Collected safely",
            "Sorted at recycling facility",
            "Processed for material recovery"
        ]

    if not result["recyclability"]:
        result["recyclability"] = [
            "Contains recyclable materials",
            "Some components can be reused",
            "Proper recycling improves recovery efficiency"
        ]

    if not result["environmental_impact"]:
        result["environmental_impact"] = [
            "Improper disposal can harm environment",
            "Toxic elements may pollute soil",
            "Recycling reduces environmental damage"
        ]

    # Dynamic safe scores
    try:
        result["recyclability_score"] = int(result.get("recyclability_score"))
    except:
        result["recyclability_score"] = 70

    try:
        result["impact_score"] = int(result.get("impact_score"))
    except:
        result["impact_score"] = 70

    result.setdefault("identified_as", "Electronic Device")

    return result


# ---------- IMAGE AI ----------
def ai_detect_e_waste(image_path):
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in e-waste recycling.\n"
                    "Respond ONLY in valid JSON.\n\n"
                    "Rules:\n"
                    "- handling_procedure must be a list\n"
                    "- recyclability must be a list of bullet points\n"
                    "- environmental_impact must be a list of bullet points\n"
                    "- recyclability_score MUST be between 60 and 95\n"
                    "- impact_score MUST be between 60 and 95\n\n"
                    "Return keys:\n"
                    "identified_as, handling_procedure, recyclability, environmental_impact, recyclability_score, impact_score"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this e-waste image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                ]
            }
        ],
        temperature=0.4
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    return clean_result(json.loads(raw))


# ---------- TEXT AI ----------
def ai_explain_known_e_waste(label):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Respond ONLY in JSON with lists and dynamic scores"
            },
            {
                "role": "user",
                "content": f"""
Item: {label}

Return JSON:
identified_as,
handling_procedure (list),
recyclability (list),
environmental_impact (list),
recyclability_score (60–95),
impact_score (60–95)
"""
            }
        ],
        temperature=0.4
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    result = json.loads(raw)
    result["identified_as"] = label

    return clean_result(result)