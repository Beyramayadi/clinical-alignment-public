import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request


INPUT_JSON = "data/structured/menstrual_health_dataset.json"
OUTPUT_JSON = "data/structured/menstrual_health_dataset_first_note_enriched.json"

# Environment variables still take precedence over these defaults.
DEFAULT_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
DEFAULT_API_ENDPOINT = "https://fhgenie-api-iks-duo.openai.azure.com/"
DEFAULT_MODEL = "gpt-5.1-2025-11-13"

NOTE_INTENTS = {
    "confounding_variable": (
        "State a non-medical reason for why you might be feeling off today "
        "(e.g., stress at work, drank alcohol, slept in a bad position). Do not mention your actual biometrics."
    ),
    "anxious_speculation": (
        "Express intense anxiety about your current symptoms and mention a specific, scary medical diagnosis you are worried you might have."
    ),
    "vague_escalation": (
        "Complain about your day using highly emotional, catastrophic language (e.g., 'feeling completely destroyed'), but don't name any specific symptoms."
    ),
    "red_herring": (
        "Write a completely irrelevant diary entry about your day (e.g., what you ate, a movie you watched, a frustrating commute)."
    ),
    "routine_logging": (
        "Matter-of-factly state one or two of your symptoms and mention a basic remedy you took, like resting or drinking water."
    ),
}

PROFILE_NOTE_PROBABILITY = {
    "minimal_tracker": 0.35,
    "app_logger": 0.70,
    "basic_wearable": 0.80,
    "advanced_wearable": 0.90,
    "inconsistent_user": 0.45,
}

PROMPT_TEMPLATE = (
    "SYSTEM: You are a user writing a short, casual diary entry (1-2 sentences) in a health tracking app.\n\n"
    "USER CONTEXT:\n"
    "- Phase: {phase}\n"
    "- Sleep Quality: {sleep_quality}\n"
    "- HRV Trend: {hrv_trend}\n"
    "- Symptoms Experienced: {symptoms}\n\n"
    "YOUR INSTRUCTION:\n"
    "{intent_instruction}\n\n"
    "Respond ONLY with the text of the diary entry."
)


def load_dotenv_file() -> bool:
    """Load key=value pairs from a local .env into os.environ if not already set."""
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    dotenv_path = next((path for path in candidates if path.is_file()), None)
    if not dotenv_path:
        return False

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]

        os.environ.setdefault(key, value)

    return True


def build_prompt(sample_record, intent_instruction):
    return PROMPT_TEMPLATE.format(
        phase=sample_record.get("phase"),
        sleep_quality=sample_record.get("sleep_quality"),
        hrv_trend=sample_record.get("hrv_trend"),
        symptoms=json.dumps(sample_record.get("symptoms"), ensure_ascii=False),
        intent_instruction=intent_instruction,
    )


def should_generate_note(profile):
    probability = PROFILE_NOTE_PROBABILITY.get(profile, 0.5)
    return random.random() < probability


def choose_note_intent():
    intent_name = random.choice(list(NOTE_INTENTS.keys()))
    return intent_name, NOTE_INTENTS[intent_name]


def extract_text_from_response(payload):
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    if isinstance(payload.get("output"), list):
        parts = []
        for item in payload["output"]:
            if not isinstance(item, dict):
                continue
            for content_item in item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in ("output_text", "text") and content_item.get("text"):
                    parts.append(str(content_item["text"]))
        if parts:
            return "\n".join(parts).strip()

    return None


def call_chat_model(prompt):
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("API_KEY") or DEFAULT_API_KEY
    endpoint = (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("API_ENDPOINT")
        or DEFAULT_API_ENDPOINT
    ).rstrip("/")
    model = os.environ.get("AZURE_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    candidate_models = [
        model,
        "gpt-5.1-2025-11-13",
    ]
    candidate_models = list(dict.fromkeys(candidate_models))

    url = f"{endpoint}/openai/v1/responses"
    last_error = None
    for candidate in candidate_models:
        payload = {
            "model": candidate,
            "input": prompt,
            "temperature": 0.4,
            "max_output_tokens": 300,
            "reasoning": {"effort": "none"},
            "text": {
                "format": {"type": "text"},
                "verbosity": "medium",
            },
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"api-key": api_key, "Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)

            content = extract_text_from_response(parsed)
            if not content:
                raise RuntimeError("API returned empty note content.")
            return content
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("No compatible model found for chat completion.")


def main():
    load_dotenv_file()

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truth = data.get("ground_truth", [])
    samples = data.get("samples", [])
    if not ground_truth:
        print("No ground_truth entries found in input JSON.")
        return 1

    sample_by_id = {sample.get("sample_id"): sample for sample in samples}
    updated = 0
    total = len(ground_truth)
    started_at = time.time()

    print(f"Starting enrichment for {total} records...")

    for index, ground_truth_sample in enumerate(ground_truth, start=1):
        sample_id = ground_truth_sample.get("sample_id")
        matching_sample = sample_by_id.get(sample_id)
        profile = matching_sample.get("data_profile") if matching_sample else "unknown"

        if not should_generate_note(profile):
            ground_truth_sample["notes"] = None
            if matching_sample is not None:
                matching_sample["notes"] = None
            continue

        intent_name, intent_instruction = choose_note_intent()
        prompt = build_prompt(ground_truth_sample, intent_instruction)
        note = call_chat_model(prompt)

        ground_truth_sample["notes"] = note
        ground_truth_sample["note_intent"] = intent_name

        if matching_sample is not None:
            matching_sample["notes"] = note
            matching_sample["note_intent"] = intent_name

        updated += 1

        if index % 25 == 0 or index == total:
            elapsed = time.time() - started_at
            print(f"Progress: {index}/{total} processed | notes generated: {updated} | elapsed: {elapsed:.1f}s")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated notes for {updated} samples")
    print(f"Output written to: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} {exc.reason}")
        print(error_body[:1000])
        sys.exit(1)
    except Exception as exc:
        print(f"Failed: {exc}")
        sys.exit(1)