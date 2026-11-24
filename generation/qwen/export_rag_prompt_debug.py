import json
from pathlib import Path
import time

import generate_patient_summaries_rag_grounded as rag


INPUT_JSON = rag.INPUT_JSON
README_CSV = rag.README_CSV
OUTPUT_JSON = "data/generated_summaries/qwen/rag_prompt_debug.json"


def dedupe_matches(matches):
    deduped = []
    seen = set()
    for item in matches:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        signature = (term.lower(), definition.lower())
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def main() -> int:
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    rag_index = rag.RagIndex(README_CSV)
    if rag_index.is_ready:
        print(
            f"RAG index ready: {rag_index.df.shape[0]} rows, "
            f"vocab size {len(rag_index.vectorizer.vocabulary_)}."
        )
    else:
        print("RAG index is empty. Prompts will contain fallback context text.")

    with input_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        print("Expected the input JSON to be a list of sample objects.")
        return 1

    debug_rows = []
    total = len(samples)
    started_at = time.time()

    print(f"Exporting prompt debug rows for {total} samples...")

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            continue

        sample_id = sample.get("sample_id")
        if not sample_id:
            continue

        sample_for_debug = dict(sample)
        sample_for_debug.pop("notes", None)

        prompt, matches = rag.build_prompt(sample_for_debug, rag_index)
        matches = dedupe_matches(matches)

        debug_rows.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "retrieved_context_used": matches,
                "retrieved_count": len(matches),
            }
        )

        if index % 25 == 0 or index == total:
            elapsed = time.time() - started_at
            pct = (index / total) * 100 if total else 100.0
            print(
                f"Progress: {index}/{total} ({pct:.1f}%) | rows: {len(debug_rows)} | elapsed: {elapsed:.1f}s"
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(debug_rows, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(debug_rows)} debug rows to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
