import json
import os
from pathlib import Path
import sys
import time
import urllib.error

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ---------------------------------------------------------------------------
# File Configurations
# ---------------------------------------------------------------------------
INPUT_JSON = "data/structured/menstrual_health_dataset_clean_samples.json"
OUTPUT_JSON = "data/generated_summaries/phi/menstrual_health_patient_summaries_rag_constrained.json"
README_CSV = "readme_menstrual_grounded.csv"  # grounded CSV: columns ann_text, gpt_generated

# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_API_KEY = "ollama"
DEFAULT_LOCAL_MODEL = "phi4-mini:3.8b"

# ---------------------------------------------------------------------------
# Retrieval Configurations
# ---------------------------------------------------------------------------
TOP_K = 3                 # max number of grounding snippets injected per sample
MIN_SIMILARITY = 0.12      # cosine similarity floor; below this, we treat it as "no match"
TEXT_COL = "ann_text"
DEFINITION_COL = "gpt_generated"

PROMPT_TEMPLATE = (
    "SYSTEM: You are a highly precise clinical summarization assistant. Your task is to translate "
    "structured, longitudinal physiological data into an accessible, lay-language patient narrative. "
    "You must strictly follow the constraints below.\n\n"
    "### INPUT DATA\n"
    "- Structured Physiological Record (X):\n"
    "{user_data}\n\n"
    "### RETRIEVED CONTEXT (RAG GROUNDING)\n"
    "Use this context only as support for explanations and wording. Do not contradict the structured record.\n"
    "{retrieved_context}\n\n"
    "### STAGE-GATED GENERATION CONSTRAINTS\n"
    "1. Evidence Consistency: Only write claims directly supported by the provided record and retrieved context. "
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
    "- Symptom Context\n"
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


class RagIndex:
    """
    Real similarity-based retrieval over the grounded CSV, using TF-IDF + cosine
    similarity instead of exact string matching.

    Why TF-IDF instead of an embedding model: it's fully local (no model download,
    no extra network dependency), fast to build/query at this corpus size, and a
    clear upgrade path exists (swap in sentence-transformers or an API embedding
    call later without changing the calling code below).
    """

    def __init__(self, csv_path: str, text_col: str = TEXT_COL, def_col: str = DEFINITION_COL,
                 top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY):
        self.text_col = text_col
        self.def_col = def_col
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.df = pd.DataFrame()
        self.vectorizer = None
        self.doc_matrix = None

        path = Path(csv_path)
        if not path.is_file():
            print(f"⚠️ Warning: RAG database file not found at {csv_path}. RAG will be empty!")
            return

        df = pd.read_csv(path)
        missing_cols = [c for c in (text_col, def_col) if c not in df.columns]
        if missing_cols:
            print(f"⚠️ Warning: CSV missing expected columns {missing_cols}. RAG will be empty!")
            return

        df = df.dropna(subset=[text_col, def_col]).reset_index(drop=True)
        # Remove duplicate grounding rows early to reduce repeated retrieval hits.
        df["_term_norm"] = df[text_col].astype(str).str.strip().str.lower()
        df["_def_norm"] = df[def_col].astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=["_term_norm", "_def_norm"]).reset_index(drop=True)
        df = df.drop(columns=["_term_norm", "_def_norm"])
        if df.empty:
            print("⚠️ Warning: RAG database has no usable rows. RAG will be empty!")
            return

        self.df = df
        # ngram_range=(1, 2) lets "menstrual cramps" match "cramps" reasonably well
        # via shared unigram terms, while still capturing some phrase-level signal.
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_matrix = self.vectorizer.fit_transform(self.df[text_col].astype(str))

    @property
    def is_ready(self) -> bool:
        return self.doc_matrix is not None and self.doc_matrix.shape[0] > 0

    def retrieve(self, query_text: str):
        """Return up to top_k (term, definition, score) matches above the similarity floor."""
        if not self.is_ready or not query_text.strip():
            return []

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.doc_matrix).flatten()

        ranked_idx = sims.argsort()[::-1]
        results = []
        seen_signatures = set()
        for idx in ranked_idx[: self.top_k * 4]:  # look a bit past top_k before filtering
            score = float(sims[idx])
            if score < self.min_similarity:
                break  # sorted descending, so nothing further will pass either

            term = str(self.df.iloc[idx][self.text_col]).strip()
            definition = str(self.df.iloc[idx][self.def_col]).strip()
            signature = (term.lower(), definition.lower())
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            results.append({
                "term": term,
                "definition": definition,
                "score": round(score, 4),
            })
            if len(results) >= self.top_k:
                break
        return results


def build_query_text(sample_record: dict) -> str:
    """
    Generic, field-agnostic query builder: turns every field in the record into
    searchable text, rather than hardcoding a fixed list of keys (phase, symptoms,
    hrv_trend). New fields in the dataset are picked up automatically.
    """
    parts = []
    for key, value in sample_record.items():
        if key in {"sample_id", "notes"}:
            continue

        # Include the key name itself as signal (e.g. "hrv_trend" -> "hrv trend")
        parts.append(key.replace("_", " "))

        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(v) for v in value)
        elif isinstance(value, dict):
            for k, v in value.items():
                parts.append(str(k).replace("_", " "))
                parts.append(str(v))
        elif isinstance(value, (int, float, bool)):
            # Numeric values alone aren't useful lexical signal for TF-IDF matching,
            # so we skip the raw number but keep the field name above.
            continue

    return " ".join(str(p) for p in parts if str(p).strip())


def format_retrieved_context(matches) -> str:
    if not matches:
        return "No specific medical definitions retrieved for today's metrics."
    lines = [f"- {m['term']}: {m['definition']}" for m in matches]
    return "\n".join(lines)


def build_prompt(sample_record: dict, rag_index: RagIndex):
    model_input = dict(sample_record)
    model_input.pop("sample_id", None)
    model_input.pop("notes", None)

    query_text = build_query_text(sample_record)
    matches = rag_index.retrieve(query_text)
    retrieved_context = format_retrieved_context(matches)

    user_data = json.dumps(model_input, ensure_ascii=False)

    prompt = PROMPT_TEMPLATE.format(
        retrieved_context=retrieved_context,
        user_data=user_data,
    )
    return prompt, matches


def create_client_and_model():
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LOCAL_OPENAI_BASE_URL") or DEFAULT_LOCAL_BASE_URL
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_OPENAI_API_KEY") or DEFAULT_LOCAL_API_KEY
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("LOCAL_OPENAI_MODEL") or DEFAULT_LOCAL_MODEL
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def call_summary_model(prompt: str, max_retries: int = 3, backoff_seconds: float = 2.0):
    """Calls the model with simple retry/backoff so a transient failure doesn't kill the whole batch."""
    client, model = create_client_and_model()

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a clinical-to-lay translation system. Do not deviate from constraints."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=450,  # increased headroom for the 3-section structured output
            )

            text = None
            if response and response.choices and response.choices[0].message:
                text = response.choices[0].message.content
            if not text or not str(text).strip():
                raise RuntimeError("Model returned empty summary content.")
            return str(text).strip()

        except Exception as exc:  # noqa: BLE001 - we want to retry on any transient failure
            last_error = exc
            if attempt < max_retries:
                wait = backoff_seconds * attempt
                print(f"  ⚠️ Model call failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise last_error


def main():
    load_dotenv_file()

    rag_index = RagIndex(README_CSV)
    if rag_index.is_ready:
        print(f"RAG index ready: {rag_index.df.shape[0]} rows, TF-IDF vocab size "
              f"{len(rag_index.vectorizer.vocabulary_)}.")
    else:
        print("RAG index is empty — all samples will fall back to 'no context retrieved'.")

    try:
        progress_every = max(1, int(os.environ.get("PROGRESS_EVERY", "10")))
    except ValueError:
        progress_every = 10

    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Coverage tracking so grounding quality is measurable, not invisible.
    samples_with_zero_matches = 0
    samples_processed = 0
    match_count_total = 0

    print(f"Starting patient-summary generation with RAG-grounding for {total} samples...")

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            continue

        sample_id = sample.get("sample_id")
        if not sample_id:
            continue

        prompt, matches = build_prompt(sample, rag_index)

        try:
            summary = call_summary_model(prompt)
        except Exception as exc:
            print(f"  ❌ Failed to generate summary for sample {sample_id}: {exc}. Skipping.")
            continue

        samples_processed += 1
        match_count_total += len(matches)
        if not matches:
            samples_with_zero_matches += 1

        summaries.append({
            "sample_id": sample_id,
            "patient_summary": summary,
            # Traceability: exactly what was retrieved and injected for this sample,
            # so grounding can be audited after the fact instead of trusted blindly.
            "retrieved_context_used": matches,
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

    if samples_processed:
        zero_match_pct = (samples_with_zero_matches / samples_processed) * 100
        avg_matches = match_count_total / samples_processed
        print(
            "Retrieval coverage: "
            f"{samples_with_zero_matches}/{samples_processed} samples ({zero_match_pct:.1f}%) had NO grounding matches | "
            f"avg matches/sample: {avg_matches:.2f}"
        )

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