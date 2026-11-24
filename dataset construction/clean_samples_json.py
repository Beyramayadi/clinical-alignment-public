import argparse
import json
from pathlib import Path


DEFAULT_INPUT = "data/structured/menstrual_health_dataset_first_note_enriched.json"
DEFAULT_OUTPUT = "data/structured/menstrual_health_dataset_clean_samples.json"

# Fields to remove from each sample record.
FIELDS_TO_DROP = {
    "data_profile",
    "available_fields",
    "missing_fields",
    "note_intent",
}


def clean_samples(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples", [])
    cleaned = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        cleaned_sample = {
            key: value
            for key, value in sample.items()
            if key not in FIELDS_TO_DROP
        }
        # Keep sample_id by default; skip malformed records that do not have it.
        if "sample_id" in cleaned_sample:
            cleaned.append(cleaned_sample)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Input samples: {len(samples)}")
    print(f"Cleaned samples written: {len(cleaned)}")
    print(f"Output file: {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a clean JSON file containing only sample records "
            "from the enriched menstrual health dataset."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to source JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to output JSON file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    try:
        return clean_samples(input_path, output_path)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in input file: {exc}")
        return 1
    except Exception as exc:
        print(f"Failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
