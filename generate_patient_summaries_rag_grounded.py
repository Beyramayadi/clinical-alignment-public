import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import pandas as pd  # <-- Required for RAG matching
from openai import OpenAI

# File Configurations
INPUT_JSON = "menstrual_health_dataset_clean_samples.json"
OUTPUT_JSON = "menstrual_health_patient_summaries.json"
README_CSV = "readme_menstrual_grounded.csv"  # <-- Your grounded CSV

# Model Configurations
DEFAULT_MODEL = "gpt-5.1-2025-11-13"
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_API_KEY = "ollama"
DEFAULT_LOCAL_MODEL = "phi4-mini:3.8b"

# Your strict, zero-drift prompt template
PROMPT_TEMPLATE = (
    "SYSTEM: You are an expert, empathetic clinical communicator. Translate complex personal "
    "health data into accessible, supportive, and safe patient narratives. You must strictly "
    "prevent semantic drift by adhering to the safety constraints below.\n\n"
    
    "STRICT GENERATION CONSTRAINTS:\n"
    "1. Evidence Consistency: Every statement must be directly supported by, and traceable to, "
    "the [USER DATA] or [RETRIEVED CONTEXT]. Do not extrapolate trends for biometrics listed in 'missing_fields'.\n"
    "2. Uncertainty Preservation: Match the confidence level of the narrative to the 'confidence_score' and 'missing_fields'. "
    "If confidence is low, use hedged language (e.g., 'may suggest', 'might be', 'appears to')[cite: 1].\n"
    "3. Diagnostic Constraint: Limit descriptions to observed physiological trends and symptoms (e.g., elevated temp, cramps)[cite: 1]. "
    "Strictly avoid implying or mentioning clinical diagnoses (like Endometriosis, PCOS, or thyroid issues)[cite: 1].\n"
    "4. Nuance Preservation: Represent the complete tracking day. Briefly explain what missing fields mean for clarity[cite: 1].\n"
    "5. Urgency Calibration: Keep tone supportive and neutral. Match tone severity to the actual biometric numbers[cite: 1].\n\n"
    
    "[RETRIEVED CONTEXT] (Definitions & Explanations):\n"
    "{retrieved_context}\n\n"
    
    "[USER DATA]:\n"
    "{user_data}\n\n"
    "Return only the patient-facing summary text (maximum 3 sentences/points)."
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


def load_rag_database(csv_path: str) -> pd.DataFrame:
    """Loads the grounded CSV to use as our RAG search index."""
    path = Path(csv_path)
    if not path.is_file():
        print(f"⚠️ Warning: RAG database file not found at {csv_path}. RAG will be empty!")
        return pd.DataFrame()
    
    # Load and normalize the search column to lowercase
    df = pd.read_csv(path)
    df["ann_text_lower"] = df["ann_text"].astype(str).str.strip().str.lower()
    return df


def retrieve_context_for_record(sample_record: dict, rag_df: pd.DataFrame) -> str:
    """
    Extracts terms (symptoms, phases, biometrics) from the structured record,
    queries the RAG DataFrame, and formats a grounding block of definitions.
    """
    if rag_df.empty:
        return "No retrieved context available."

    # Step 1: Identify keywords in the patient record to search for
    keywords_to_search = []
    
    # Add phase (e.g. 'luteal', 'follicular')[cite: 1]
    if "phase" in sample_record and isinstance(sample_record["phase"], str):
        keywords_to_search.append(sample_record["phase"])
    
    # Add explicit symptom strings (e.g. 'fatigue', 'cramps')[cite: 1]
    if "symptoms" in sample_record and isinstance(sample_record["symptoms"], list):
        keywords_to_search.extend(sample_record["symptoms"])
        
    # Always include biometric terms we want to ensure are contextualized[cite: 1]
    if "hrv_trend" in sample_record:
        keywords_to_search.extend(["hrv", "heart rate variability"])

    # Step 2: Query the CSV for those terms (case-insensitive)
    retrieved_lines = []
    seen_terms = set()
    
    for term in keywords_to_search:
        term_clean = str(term).strip().lower()
        if term_clean in seen_terms:
            continue
        seen_terms.add(term_clean)
        
        # Match against normalized database column
        matches = rag_df[rag_df["ann_text_lower"] == term_clean]
        if not matches.empty:
            # We grab the lay definition from the 'gpt_generated' column[cite: 1]
            lay_definition = matches.iloc[0]["gpt_generated"]
            retrieved_lines.append(f"- {term.capitalize()}: {lay_definition}")
            
    if not retrieved_lines:
        return "No specific medical definitions retrieved for today's metrics."
        
    return "\n".join(retrieved_lines)


def build_prompt(sample_record, rag_df):
    model_input = dict(sample_record)
    model_input.pop("sample_id", None)
    
    # 1. Dynamically retrieve lay definitions matching today's record[cite: 1]
    retrieved_context = retrieve_context_for_record(sample_record, rag_df)
    
    # 2. Stringify raw physiological data[cite: 1]
    user_data = json.dumps(model_input, ensure_ascii=False)
    
    # 3. Format everything cleanly into our system template[cite: 1]
    return PROMPT_TEMPLATE.format(
        retrieved_context=retrieved_context,
        user_data=user_data
    )


def create_client_and_model():
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LOCAL_OPENAI_BASE_URL") or DEFAULT_LOCAL_BASE_URL
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_OPENAI_API_KEY") or DEFAULT_LOCAL_API_KEY
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("LOCAL_OPENAI_MODEL") or DEFAULT_LOCAL_MODEL
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def call_summary_model(prompt):
    client, model = create_client_and_model()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a clinical-to-lay translation system. Do not deviate from constraints[cite: 1]."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2, # Keep low to prevent random hallucinations
        max_tokens=300,
    )

    text = None
    if response and response.choices and response.choices[0].message:
        text = response.choices[0].message.content
    if not text or not str(text).strip():
        raise RuntimeError("Model returned empty summary content.")
    return str(text).strip()


def main():
    load_dotenv_file()
    
    # Load RAG Dataframe[cite: 1]
    rag_df = load_rag_database(README_CSV)
    
    try:
        progress_every = max(1, int(os.environ.get("PROGRESS_EVERY", "10")))
    except ValueError:
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

    print(f"Starting patient-summary generation with RAG-grounding for {total} samples...")

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            continue

        sample_id = sample.get("sample_id")
        if not sample_id:
            continue

        # Build prompt with matched RAG definitions from CSV[cite: 1]
        prompt = build_prompt(sample, rag_df)
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