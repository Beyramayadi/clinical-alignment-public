import json
from pathlib import Path
import sys
import time
import urllib.error
from openai import OpenAI


INPUT_JSON = "data/structured/menstrual_health_dataset_clean_samples.json"
OUTPUT_JSON = "data/generated_summaries/phi/menstrual_health_patient_summaries_constrained.json"
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_API_KEY = "ollama"
DEFAULT_LOCAL_MODEL = "phi4-mini:3.8b"

PROMPT_TEMPLATE = (
    "SYSTEM: You are a highly precise clinical summarization assistant. Your task is to translate "
    "structured, longitudinal physiological data into an accessible, lay-language patient narrative. "
    "You must strictly follow the constraints below.\n\n"
    "### INPUT DATA\n"
    "- Structured Physiological Record (X):\n"
    "{user_data}\n\n"
    "### STAGE-GATED GENERATION CONSTRAINTS\n"
    "1. Evidence Consistency: Only write claims directly supported by the provided record. "
    "Do not mention, assume, or infer metrics that are missing or absent.\n"
    "2. Uncertainty Preservation: Use hedged language when confidence is low or evidence is incomplete. "
    "Do not inflate certainty.\n"
    "3. Diagnostic Restraint: Do not infer or suggest any diagnosis, disease, or pathology. "
    "Describe only observed physiology and reported symptoms.\n"
    "4. Nuance Preservation: Keep important qualifiers like phase context, confidence, and explicit trends.\n"
    "5. Tone Calibration: Keep tone supportive and balanced. Avoid panic and avoid over-reassurance.\n\n"
    "### OUTPUT FORMAT\n"
    "Return only a patient-facing summary in clear, empathetic, lay language with:\n"
    "- Your Current Status\n"
    "- What your data is showing\n"
    
)


def build_prompt(sample_record):
    model_input = dict(sample_record)
    model_input.pop("sample_id", None)
    user_data = json.dumps(model_input, ensure_ascii=False)
    return PROMPT_TEMPLATE.format(user_data=user_data)


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


def create_client_and_model():
    client = OpenAI(base_url=DEFAULT_LOCAL_BASE_URL, api_key=DEFAULT_LOCAL_API_KEY)
    return client, DEFAULT_LOCAL_MODEL


def call_summary_model(prompt):
    client, model = create_client_and_model()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful health assistant. Return only the summary text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=250,
    )

    text = None
    if response and response.choices and response.choices[0].message:
        text = response.choices[0].message.content
    if not text or not str(text).strip():
        raise RuntimeError("Model returned empty summary content.")
    return str(text).strip()


def main():
    progress_every = 10

    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_JSON)

    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    with input_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        print("Expected the input JSON to be a list of sample objects.")
        return 1

    summaries = []
    total = len(samples)
    started_at = time.time()

    print(f"Starting patient-summary generation for {total} samples (progress every {progress_every} rows)...")

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            continue

        sample_id = sample.get("sample_id")
        if not sample_id:
            continue

        prompt = build_prompt(sample)
        summary = call_summary_model(prompt)

        summaries.append({
            "sample_id": sample_id,
            "patient_summary": summary,
        })

        if index % progress_every == 0 or index == total:
            elapsed = time.time() - started_at
            pct = (index / total) * 100 if total else 100.0
            print(
                f"Progress: {index}/{total} ({pct:.1f}%) | summaries: {len(summaries)} | elapsed: {elapsed:.1f}s"
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(summaries)} patient summaries to: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} {exc.reason}")
        print(error_body[:1000])
        raise SystemExit(1)
    except Exception as exc:
        print(f"Failed: {exc}")
        raise SystemExit(1)
