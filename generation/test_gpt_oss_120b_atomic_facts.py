import json
import time
from pathlib import Path

from openai import OpenAI

from atomic_facts import (
    DEFAULT_LOCAL_API_KEY,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    FEW_SHOT_EXAMPLES,
    PROMPT_PLACEHOLDER,
)


TARGET_JSON = Path("data/generated_summaries/phi/menstrual_health_patient_summaries.json")


def load_first_summary() -> str:
    with TARGET_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Expected a non-empty list in {TARGET_JSON}")

    summary = data[0].get("patient_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("First record does not contain a usable patient_summary")
    return summary.strip()


def create_client() -> OpenAI:
    return OpenAI(base_url=DEFAULT_LOCAL_BASE_URL, api_key=DEFAULT_LOCAL_API_KEY)


def run_probe(label: str, messages, max_tokens: int) -> None:
    client = create_client()
    started_at = time.perf_counter()
    response = client.chat.completions.create(
        model=DEFAULT_LOCAL_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - started_at

    message = response.choices[0].message
    reasoning = getattr(message, "reasoning", None)

    print(f"=== {label} ===")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Reasoning chars: {len(reasoning) if reasoning else 0}")
    print("Content:")
    print(message.content or "<EMPTY>")
    print()


def main() -> int:
    summary = load_first_summary()

    run_probe(
        label="Simple probe",
        messages=[
            {"role": "system", "content": "You extract atomic facts. Respond only with bullet points."},
            {"role": "user", "content": "Input:\nYou are on day 10 of your cycle. Your resting heart rate is 60 bpm.\n\nOutput:"},
        ],
        max_tokens=200,
    )

    full_prompt = FEW_SHOT_EXAMPLES.replace(PROMPT_PLACEHOLDER, summary)
    run_probe(
        label="Full atomic-facts prompt on first summary",
        messages=[
            {
                "role": "system",
                "content": "You decompose a clinical or patient summary into independent atomic facts. Output only bullet points.",
            },
            {"role": "user", "content": full_prompt},
        ],
        max_tokens=2000,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())