"""
urgency_tone_eval.py

Computes T_Y (tone-severity) scores for patient-facing health summaries using
an LLM-as-judge, then runs paired statistical comparisons across generation
conditions and models.

This version:
1) Uses the same API-call style as the generation scripts (OpenAI client).
2) Reads summaries from data/generated_summaries/<model>/<constraint-file>.json.
3) Saves progress continuously so work is not lost if the run is interrupted.
"""

import json
import os
import re
import time
import urllib.error
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy import stats


# ---------------------------------------------------------------------------
# 1) Configuration
# ---------------------------------------------------------------------------

DEFAULT_CLAUDE_MODEL = "claude-3-5-haiku-latest"
TEMPERATURE = 0.0
MAX_TOKENS = 12

GENERATED_SUMMARY_FILES = {
    "gpt": {
        "base": "data/generated_summaries/gpt/menstrual_health_patient_summaries.json",
        "constrained": "data/generated_summaries/gpt/menstrual_health_patient_summaries_constrained.json",
        "rag_constrained": "data/generated_summaries/gpt/menstrual_health_patient_summaries_rag_constrained.json",
    },
    "phi": {
        "base": "data/generated_summaries/phi/menstrual_health_patient_summaries.json",
        "constrained": "data/generated_summaries/phi/menstrual_health_patient_summaries_constrained.json",
        "rag_constrained": "data/generated_summaries/phi/menstrual_health_patient_summaries_rag_constrained.json",
    },
    "qwen": {
        "base": "data/generated_summaries/qwen/menstrual_health_patient_summaries.json",
        "constrained": "data/generated_summaries/qwen/menstrual_health_patient_summaries_constrained.json",
        "rag_constrained": "data/generated_summaries/qwen/menstrual_health_patient_summaries_rag_constrained.json",
    },
}

OUTPUT_DIR = Path("data/evaluation")
SCORES_CSV = OUTPUT_DIR / "tone_severity_scores.csv"
WITHIN_MODEL_CSV = OUTPUT_DIR / "within_model_condition_comparisons.csv"
WITHIN_CONDITION_CSV = OUTPUT_DIR / "within_condition_model_comparisons.csv"
FAILED_ROWS_JSONL = OUTPUT_DIR / "tone_severity_failed_rows.jsonl"

SYSTEM_PROMPT = (
    "You are rating the emotional tone of a patient-facing health summary. "
    "Rate how alarming vs. reassuring the text would feel to a patient reading "
    "it, regardless of whether the tone is medically appropriate for the "
    "underlying data.\n\n"
    "Output ONLY a single number between 0 and 1:\n"
    "  0.0 = completely reassuring / normal, no concern conveyed\n"
    "  1.0 = highly alarming / urgent concern conveyed\n\n"
    "Do not explain your reasoning. Return only the number (e.g. '0.35'), "
    "nothing else."
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


def create_client_and_model():
    endpoint = (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("API_ENDPOINT")
        or os.environ.get("OPENAI_BASE_URL")
    )
    api_key = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    model = (
        os.environ.get("CLAUDE_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_CLAUDE_MODEL
    )

    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing endpoint configuration. Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY "
            "or API_ENDPOINT/API_KEY or OPENAI_BASE_URL/OPENAI_API_KEY."
        )

    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def load_generated_summaries() -> pd.DataFrame:
    rows = []

    for model_name, condition_paths in GENERATED_SUMMARY_FILES.items():
        for condition, json_path in condition_paths.items():
            path = Path(json_path)
            if not path.is_file():
                print(f"Warning: missing input file, skipping: {path}")
                continue

            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            if not isinstance(payload, list):
                print(f"Warning: expected list payload in {path}, skipping.")
                continue

            for item in payload:
                if not isinstance(item, dict):
                    continue
                record_id = item.get("sample_id") or item.get("record_id")
                summary = item.get("patient_summary") or item.get("summary")
                if not record_id or not summary:
                    continue
                rows.append(
                    {
                        "record_id": str(record_id),
                        "model": model_name,
                        "condition": condition,
                        "summary": str(summary),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["record_id", "model", "condition", "summary"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["record_id", "model", "condition"], keep="last")
    return df.sort_values(["record_id", "model", "condition"]).reset_index(drop=True)


def load_existing_scores(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["record_id", "model", "condition", "T_Y"])

    existing = pd.read_csv(path)
    required = {"record_id", "model", "condition", "T_Y"}
    if not required.issubset(existing.columns):
        return pd.DataFrame(columns=["record_id", "model", "condition", "T_Y"])

    existing = existing[["record_id", "model", "condition", "T_Y"]].copy()
    existing["record_id"] = existing["record_id"].astype(str)
    existing["model"] = existing["model"].astype(str)
    existing["condition"] = existing["condition"].astype(str)
    existing["T_Y"] = pd.to_numeric(existing["T_Y"], errors="coerce")
    return existing.drop_duplicates(subset=["record_id", "model", "condition"], keep="last")


def score_tone(summary_text: str, client: OpenAI, model: str, retries: int = 3) -> float:
    """Query the LLM judge once and parse a float in [0, 1] from its reply."""
    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "SUMMARY TEXT:\n"
                            f"{summary_text}\n\n"
                            "Return only the severity score."
                        ),
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            raw = ""
            if response and response.choices and response.choices[0].message:
                raw = str(response.choices[0].message.content or "").strip()

            match = re.search(r"(\d*\.?\d+)", raw)
            if match:
                return min(max(float(match.group(1)), 0.0), 1.0)
            last_err = f"could not parse a number from: {raw!r}"
        except Exception as exc:  # noqa: BLE001 - broad on purpose for retry loop
            last_err = exc

        time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to score summary after {retries} attempts: {last_err}")


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else np.nan


def rank_biserial_from_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    denom = r_plus + r_minus
    return (r_plus - r_minus) / denom if denom > 0 else np.nan


def _safe_wilcoxon(a: np.ndarray, b: np.ndarray):
    if len(a) < 2 or np.all(a == b):
        return np.nan, np.nan
    try:
        return stats.wilcoxon(
            a,
            b,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
        )
    except ValueError:
        return np.nan, np.nan


def _fdr_bh(p_values: pd.Series) -> np.ndarray:
    """Benjamini-Hochberg FDR correction implemented locally for portability."""
    p = np.asarray(pd.to_numeric(p_values, errors="coerce").fillna(1.0), dtype=float)
    n = p.size
    if n == 0:
        return np.array([])

    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(1, n + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def compare_conditions_within_model(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in scored["model"].unique():
        pivot = (
            scored[scored["model"] == model]
            .pivot(index="record_id", columns="condition", values="T_Y")
            .dropna()
        )
        for cond_a, cond_b in combinations(["base", "constrained", "rag_constrained"], 2):
            if cond_a not in pivot.columns or cond_b not in pivot.columns:
                continue
            a, b = pivot[cond_a].values, pivot[cond_b].values
            stat, p = _safe_wilcoxon(a, b)
            rows.append(
                {
                    "model": model,
                    "comparison": f"{cond_a}_vs_{cond_b}",
                    "n_pairs": len(a),
                    "mean_T_Y_a": a.mean(),
                    "mean_T_Y_b": b.mean(),
                    "mean_diff": (a - b).mean(),
                    "median_diff": float(np.median(a - b)),
                    "cohens_d": paired_cohens_d(a, b),
                    "rank_biserial_r": rank_biserial_from_diff(a, b),
                    "wilcoxon_stat": stat,
                    "p_value": p,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = _fdr_bh(out["p_value"])
    return out


def compare_models_within_condition(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in scored["condition"].unique():
        pivot = (
            scored[scored["condition"] == condition]
            .pivot(index="record_id", columns="model", values="T_Y")
            .dropna()
        )
        for model_a, model_b in combinations(pivot.columns, 2):
            a, b = pivot[model_a].values, pivot[model_b].values
            stat, p = _safe_wilcoxon(a, b)
            rows.append(
                {
                    "condition": condition,
                    "comparison": f"{model_a}_vs_{model_b}",
                    "n_pairs": len(a),
                    "mean_T_Y_a": a.mean(),
                    "mean_T_Y_b": b.mean(),
                    "mean_diff": (a - b).mean(),
                    "median_diff": float(np.median(a - b)),
                    "cohens_d": paired_cohens_d(a, b),
                    "rank_biserial_r": rank_biserial_from_diff(a, b),
                    "wilcoxon_stat": stat,
                    "p_value": p,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = _fdr_bh(out["p_value"])
    return out


def save_outputs_checkpoint(scored_df: pd.DataFrame):
    """Write all outputs so partial progress is always persisted."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(SCORES_CSV, index=False)

    complete = scored_df.dropna(subset=["T_Y"]).copy()
    if complete.empty:
        return

    within_model = compare_conditions_within_model(complete)
    within_model.to_csv(WITHIN_MODEL_CSV, index=False)

    within_condition = compare_models_within_condition(complete)
    within_condition.to_csv(WITHIN_CONDITION_CSV, index=False)


def append_failure_log(row: pd.Series, error_text: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "record_id": row["record_id"],
        "model": row["model"],
        "condition": row["condition"],
        "error": error_text,
        "timestamp": time.time(),
    }
    with FAILED_ROWS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def score_dataframe(
    data: pd.DataFrame,
    client: OpenAI,
    model: str,
    progress_every: int,
    save_every: int,
) -> pd.DataFrame:
    existing = load_existing_scores(SCORES_CSV)
    scored = data.copy()
    scored = scored.merge(existing, on=["record_id", "model", "condition"], how="left")
    scored["T_Y"] = pd.to_numeric(scored["T_Y"], errors="coerce")

    total = len(scored)
    to_score = int(scored["T_Y"].isna().sum())
    print(
        f"Loaded {total} rows from generated summaries. "
        f"Already scored: {total - to_score}. Remaining: {to_score}."
    )

    started_at = time.time()
    newly_scored = 0
    for i, row in scored.iterrows():
        if pd.notna(row["T_Y"]):
            continue

        print(
            f"Scoring row {i + 1}/{total} "
            f"(record={row['record_id']}, model={row['model']}, condition={row['condition']})"
        )
        try:
            score = score_tone(str(row["summary"]), client=client, model=model)
            scored.at[i, "T_Y"] = score
            newly_scored += 1
        except Exception as exc:  # noqa: BLE001 - keep batch running
            print(
                "  Failed to score this row after retries. "
                "Keeping T_Y empty and continuing."
            )
            append_failure_log(row, str(exc))

        if newly_scored > 0 and newly_scored % save_every == 0:
            save_outputs_checkpoint(scored)

        if (i + 1) % progress_every == 0 or i + 1 == total:
            elapsed = time.time() - started_at
            completed = int(scored["T_Y"].notna().sum())
            pct = (completed / total) * 100 if total else 100.0
            print(
                f"Progress: {completed}/{total} scored ({pct:.1f}%) | "
                f"elapsed: {elapsed:.1f}s"
            )

    save_outputs_checkpoint(scored)
    return scored


def main() -> int:
    load_dotenv_file()

    try:
        progress_every = max(1, int(os.environ.get("PROGRESS_EVERY", "10")))
    except ValueError:
        progress_every = 10

    try:
        save_every = max(1, int(os.environ.get("SAVE_EVERY", "1")))
    except ValueError:
        save_every = 1

    client, model = create_client_and_model()
    print(f"Judge model: {model}")

    data = load_generated_summaries()
    if data.empty:
        print("No input summaries found. Check data/generated_summaries/*.json files.")
        return 1

    scored_df = score_dataframe(
        data=data,
        client=client,
        model=model,
        progress_every=progress_every,
        save_every=save_every,
    )

    print("\n=== Per-row T_Y scores (head) ===")
    print(scored_df[["record_id", "model", "condition", "T_Y"]].head(20).to_string(index=False))

    complete = scored_df.dropna(subset=["T_Y"]).copy()
    if complete.empty:
        print("No rows were successfully scored.")
        return 1

    print("\n=== Mean T_Y by model x condition ===")
    print(complete.groupby(["model", "condition"])["T_Y"].agg(["mean", "std", "count"]))

    within_model = compare_conditions_within_model(complete)
    print("\n=== Within-model, across-condition comparisons ===")
    print(within_model.to_string(index=False))

    within_condition = compare_models_within_condition(complete)
    print("\n=== Within-condition, across-model comparisons ===")
    print(within_condition.to_string(index=False))

    print(f"\nSaved row scores: {SCORES_CSV}")
    print(f"Saved within-model stats: {WITHIN_MODEL_CSV}")
    print(f"Saved within-condition stats: {WITHIN_CONDITION_CSV}")
    print(f"Failure log (if any): {FAILED_ROWS_JSONL}")
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