import json
import os
from pathlib import Path
import re
import time
import urllib.error

from openai import OpenAI


TARGET_ROOT = Path("data/generated_summaries/gpt")
TARGET_FILE_NAME = "menstrual_health_patient_summaries_rag_constrained.json"

DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_API_KEY = "ollama"
DEFAULT_LOCAL_MODEL = "llama3.3:70b"
DEFAULT_MAX_TOKENS = 2000
TEST_FIRST_RECORD_ONLY = False

# Few-shot prompt examples kept from the original file.
FEW_SHOT_EXAMPLES = """You are an expert data extractor. Your task is to break down a clinical or patient summary into a list of "atomic facts." 

An atomic fact is a short, independent sentence that contains a single verifiable claim. 
Rules:
1. Break compound sentences into separate statements.
2. Replace pronouns (you, your, he, she) with "The patient" or "The patient's" to ensure each fact is self-contained and independently verifiable.
3. Output the facts as a bulleted list.

Here are some examples of how to do this:

Input:
You're on day 10 of your 30-day cycle, and your resting heart rate is steady at around 60 bpm—great job staying consistent! It sounds like you had a fun and spicy day, trying new food and enjoying some good laughs with old sitcoms. Keep up the good vibes!

Output:
- The patient is on day 10 of their cycle.
- The patient has a 30-day cycle length.
- The patient's resting heart rate is steady.
- The patient's resting heart rate is around 60 bpm.
- The patient had a fun day.
- The patient had a spicy day.
- The patient tried new food.
- The patient watched old sitcoms.
- The patient laughed while watching sitcoms.
- The patient has been consistent with their health tracking.

Input:
You're on day 29 of your cycle, currently in the luteal phase, with a cycle length of about 38 days. It looks like you had moderate sleep quality last night, getting around 5.7 hours of rest. Your resting heart rate is 69 bpm, and your oxygen levels are healthy at 97%. You're experiencing some breast tenderness and mood changes, which are common during this phase. Despite a longer-than-usual commute, you made the most of it by enjoying a book and a snack. Ending your day with comfort food and a cozy show sounds like a nice way to unwind—great job finding small joys in a busy day!

Output:
- The patient is on day 29 of their cycle.
- The patient is currently in the luteal phase.
- The patient has a cycle length of about 38 days.
- The patient had moderate sleep quality last night.
- The patient got around 5.7 hours of rest last night.
- The patient's resting heart rate is 69 bpm.
- The patient's oxygen levels are healthy.
- The patient's oxygen levels are at 97%.
- The patient is experiencing breast tenderness.
- The patient is experiencing mood changes.
- Breast tenderness and mood changes are common during the luteal phase.
- The patient had a longer-than-usual commute today.
- The patient read a book during their commute.
- The patient ate a snack during their commute.
- The patient ended their day with comfort food.
- The patient ended their day watching a cozy show.

Input:
[INSERT YOUR NEW PATIENT SUMMARY HERE]

Output:"""

PROMPT_PLACEHOLDER = "[INSERT YOUR NEW PATIENT SUMMARY HERE]"


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


def create_client_and_model():
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LOCAL_OPENAI_BASE_URL") or DEFAULT_LOCAL_BASE_URL
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_OPENAI_API_KEY") or DEFAULT_LOCAL_API_KEY
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("LOCAL_OPENAI_MODEL") or DEFAULT_LOCAL_MODEL
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def build_decomposition_prompt(summary_text: str) -> str:
    if PROMPT_PLACEHOLDER in FEW_SHOT_EXAMPLES:
        return FEW_SHOT_EXAMPLES.replace(PROMPT_PLACEHOLDER, summary_text)

    # Fallback for custom prompt variants without the placeholder marker.
    return f"{FEW_SHOT_EXAMPLES}\n\nInput:\n{summary_text}\n\nOutput:"


def parse_atomic_facts(raw_text: str):
    facts = []
    for line in raw_text.splitlines():
        clean = line.strip()
        if not clean:
            continue

        if clean.startswith("- "):
            fact = clean[2:].strip()
        elif clean.startswith("*"):
            fact = clean.lstrip("*").strip()
        else:
            fact = clean

        if fact:
            facts.append(fact)

    # Remove duplicates while keeping the original order.
    seen = set()
    deduped = []
    for fact in facts:
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def decompose_summary_to_atomic_facts(summary_text: str, max_retries: int = 3, backoff_seconds: float = 2.0):
    client, model = create_client_and_model()
    normalized_summary = re.sub(r"\s+", " ", summary_text.replace("\n", " ")).strip()
    prompt = build_decomposition_prompt(normalized_summary)
    max_tokens = int(os.environ.get("ATOMIC_FACTS_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You decompose a clinical or patient summary into independent atomic facts. Output only bullet points.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )

            text = None
            if response and response.choices and response.choices[0].message:
                text = response.choices[0].message.content
            if not text or not str(text).strip():
                raise RuntimeError(
                    "Model returned empty atomic-fact content. "
                    f"Try increasing ATOMIC_FACTS_MAX_TOKENS above {max_tokens}."
                )

            return parse_atomic_facts(str(text).strip())

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                wait = backoff_seconds * attempt
                print(f"    Warning: decomposition call failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise last_error


def find_target_files():
    return sorted(TARGET_ROOT.rglob(TARGET_FILE_NAME))


def write_json_file(file_path: Path, data):
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_path.replace(file_path)


def process_summary_record(record: dict):
    summary = record.get("patient_summary")
    if not isinstance(summary, str) or not summary.strip():
        record["atomic_facts"] = []
        return 0

    atomic_facts = decompose_summary_to_atomic_facts(summary)

    record["atomic_facts"] = atomic_facts
    return len(atomic_facts)


def process_file(file_path: Path, progress_every: int = 10):
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError(f"Expected list JSON in {file_path}")

    total = len(data)
    started_at = time.time()

    print(f"Processing {file_path} ({total} records)...")
    total_facts = 0
    processed_records = 0
    skipped_records = 0

    records_to_process = data[:1] if TEST_FIRST_RECORD_ONLY else data

    for index, record in enumerate(records_to_process, start=1):
        if not isinstance(record, dict):
            continue

        existing_atomic_facts = record.get("atomic_facts")
        if isinstance(existing_atomic_facts, list) and existing_atomic_facts:
            skipped_records += 1
            total_facts += len(existing_atomic_facts)
            if index % progress_every == 0 or index == len(records_to_process):
                elapsed = time.time() - started_at
                pct = (index / len(records_to_process)) * 100 if records_to_process else 100.0
                print(
                    f"  Progress: {index}/{len(records_to_process)} ({pct:.1f}%) | total facts so far: {total_facts} | "
                    f"processed: {processed_records} | skipped: {skipped_records} | elapsed: {elapsed:.1f}s"
                )
            continue

        facts_count = process_summary_record(record)
        write_json_file(file_path, data)
        total_facts += facts_count
        processed_records += 1

        if index % progress_every == 0 or index == len(records_to_process):
            elapsed = time.time() - started_at
            pct = (index / len(records_to_process)) * 100 if records_to_process else 100.0
            print(
                f"  Progress: {index}/{len(records_to_process)} ({pct:.1f}%) | total facts so far: {total_facts} | "
                f"processed: {processed_records} | skipped: {skipped_records} | elapsed: {elapsed:.1f}s"
            )

    if TEST_FIRST_RECORD_ONLY:
        print("  Test mode is ON: only the first record was processed in this file.")

    print(f"Updated {file_path} with atomic facts (total facts: {total_facts}).")


def main():
    load_dotenv_file()

    try:
        progress_every = max(1, int(os.environ.get("PROGRESS_EVERY", "10")))
    except ValueError:
        progress_every = 10

    target_files = find_target_files()
    if not target_files:
        print(f"No files found matching {TARGET_FILE_NAME} under {TARGET_ROOT}")
        return 1

    print(f"Found {len(target_files)} target file(s). Model: {os.environ.get('OPENAI_MODEL') or os.environ.get('LOCAL_OPENAI_MODEL') or DEFAULT_LOCAL_MODEL}")

    for file_path in target_files:
        process_file(file_path, progress_every=progress_every)

    print("Atomic-fact decomposition completed.")
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
