"""
Menstrual Health Synthetic Data Generator — v2.0 (Timeline-First Causal DAG)

Architecture:
    Cycle Length (Lognormal) → Luteal Length (Normal) → Phase Boundaries → Cycle Day
    → Phase Inference → Sleep Quality (Phase-driven) → Sleep Duration (Quality-driven)
    → Autonomic Biometrics [RHR, HRV] (Phase + Sleep override) → SpO2 (Independent)
    → Temperature Trend (Phase-driven) → Symptoms (Causal placeholder)
    → Notes (Placeholder) → User Profile Masking

References:
    - Bull et al., 2019: Cycle length lognormal distribution (N=600K+)
    - Fehring et al., 2006: Luteal phase consistency (Mean=12.9, SD=1.6)
    - Shilaih et al., 2018: BBT biphasic shift
    - Schmalenberger et al., 2019: HRV autonomic variation
    - Grant et al., 2020: RHR +2-5 bpm in luteal phase
    - Baker & Driver, 2007: Sleep architecture disruption in luteal/menstrual
    - Kräuchi, 2007: Thermoregulatory cascade and sleep onset

Author: Refactored from Phase-First → Timeline-First Causal DAG
"""

import json
import random
import numpy as np
from datetime import datetime, timezone
from copy import deepcopy

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# CONFIGURATION
# ============================================================

N_SAMPLES = 500

PROFILE_DISTRIBUTION = {
    "minimal_tracker": 75,
    "app_logger": 100,
    "basic_wearable": 125,
    "advanced_wearable": 125,
    "inconsistent_user": 75,
}

# Lognormal parameters for cycle length
# Target: mean ≈ 29.3, median ≈ 28, right-skewed, SD ≈ 4-5
# Lognormal(μ, σ): median = exp(μ), mean = exp(μ + σ²/2)
# μ = ln(28) ≈ 3.332, σ = 0.155 gives mean ≈ 28.3, SD ≈ 4.4
# Adjust μ slightly upward to hit 29.3 mean
LOGNORMAL_MU = 3.355  # exp(3.355) ≈ 28.65 (median)
LOGNORMAL_SIGMA = 0.155  # Right-skew control
CYCLE_LENGTH_MIN = 21
CYCLE_LENGTH_MAX = 50

# Luteal phase parameters (tight normal — biological constant)
LUTEAL_MEAN = 14.0
LUTEAL_SD = 1.5
LUTEAL_MIN = 10
LUTEAL_MAX = 17

# Menstrual bleed duration (fixed for simplicity)
MENSTRUAL_DURATION = 5

# ============================================================
# PHASE-CONDITIONAL PROBABILITY TABLES
# ============================================================

# Sleep quality probabilities by phase: P(sleep_quality | phase)
SLEEP_QUALITY_GIVEN_PHASE = {
    "menstrual": {"poor": 0.35, "moderate": 0.40, "good": 0.25},
    "follicular": {"poor": 0.08, "moderate": 0.32, "good": 0.60},
    "ovulatory": {"poor": 0.12, "moderate": 0.40, "good": 0.48},
    "luteal": {"poor": 0.32, "moderate": 0.43, "good": 0.25},
}

# Sleep duration ranges by quality: P(sleep_hours | sleep_quality)
SLEEP_HOURS_GIVEN_QUALITY = {
    "poor": {"min": 3.5, "max": 5.5, "mean": 4.5, "sd": 0.6},
    "moderate": {"min": 5.5, "max": 7.2, "mean": 6.3, "sd": 0.5},
    "good": {"min": 7.0, "max": 9.2, "mean": 7.8, "sd": 0.6},
}

# RHR baseline by phase (before sleep override): bpm
RHR_BASELINE_BY_PHASE = {
    "menstrual": {"mean": 63.0, "sd": 3.0, "min": 57, "max": 71},
    "follicular": {"mean": 60.0, "sd": 2.5, "min": 55, "max": 66},
    "ovulatory": {"mean": 64.0, "sd": 3.0, "min": 58, "max": 72},
    "luteal": {"mean": 67.0, "sd": 3.5, "min": 61, "max": 78},
}

# RHR acute stress penalty when sleep is poor
RHR_POOR_SLEEP_PENALTY = {"min": 4, "max": 8}

# HRV trend probabilities by phase: P(hrv_trend | phase)
HRV_TREND_GIVEN_PHASE = {
    "menstrual": {"increasing": 0.45, "stable": 0.40, "decreasing": 0.10, "erratic": 0.05},
    "follicular": {"increasing": 0.50, "stable": 0.40, "decreasing": 0.05, "erratic": 0.05},
    "ovulatory": {"increasing": 0.10, "stable": 0.15, "decreasing": 0.45, "erratic": 0.30},
    "luteal": {"increasing": 0.05, "stable": 0.25, "decreasing": 0.50, "erratic": 0.20},
}

# HRV override when sleep is poor (forces sympathetic dominance)
HRV_POOR_SLEEP_OVERRIDE = {"decreasing": 0.55, "erratic": 0.45}

# Temperature trend by phase
TEMP_TREND_GIVEN_PHASE = {
    "menstrual": {"declining": 0.55, "low_stable": 0.40, "elevated": 0.03, "biphasic_shift": 0.02},
    "follicular": {"declining": 0.05, "low_stable": 0.85, "elevated": 0.05, "biphasic_shift": 0.05},
    "ovulatory": {"declining": 0.02, "low_stable": 0.08, "elevated": 0.20, "biphasic_shift": 0.70},
    "luteal": {"declining": 0.03, "low_stable": 0.02, "elevated": 0.80, "biphasic_shift": 0.15},
}

# SpO2: independent of cycle, tightly bounded physiological range
SPO2_PARAMS = {"mean": 97.8, "sd": 0.8, "min": 95.0, "max": 100.0}

# Symptom causality mapping (placeholder structure for future LLM integration)
# Key insight: symptoms are DOWNSTREAM of biometrics, not independent
SYMPTOM_CAUSAL_TRIGGERS = {
    "fatigue": {"conditions": ["poor_sleep", "elevated_rhr"]},
    "headache": {"conditions": ["poor_sleep", "declining_hrv"]},
    "cramps": {"conditions": ["menstrual_phase", "luteal_late"]},
    "bloating": {"conditions": ["luteal_phase", "ovulatory_phase"]},
    "mood_changes": {"conditions": ["poor_sleep", "luteal_phase", "erratic_hrv"]},
    "breast_tenderness": {"conditions": ["luteal_phase"]},
    "back_pain": {"conditions": ["menstrual_phase"]},
    "nausea": {"conditions": ["ovulatory_phase", "poor_sleep"]},
}


# ============================================================
# CORE GENERATION: CAUSAL DAG NODES
# ============================================================

def generate_cycle_length():
    """
    Node 1: Cycle length from lognormal distribution.
    Bull et al., 2019 demonstrated right-skewed population distribution.
    Hard biological bounds: [21, 50] days.
    """
    raw = np.random.lognormal(mean=LOGNORMAL_MU, sigma=LOGNORMAL_SIGMA)
    clamped = int(round(np.clip(raw, CYCLE_LENGTH_MIN, CYCLE_LENGTH_MAX)))
    return clamped


def generate_luteal_phase_length():
    """
    Node 2: Luteal phase from tight normal.
    The luteal phase is the 'biological constant' — variance in total cycle
    length is almost entirely attributable to the follicular phase.
    Fehring et al., 2006: Mean=12.9d, SD=1.6d
    """
    raw = np.random.normal(loc=LUTEAL_MEAN, scale=LUTEAL_SD)
    clamped = int(round(np.clip(raw, LUTEAL_MIN, LUTEAL_MAX)))
    return clamped


def infer_phase_boundaries(cycle_length, luteal_length):
    """
    Node 3: Derive phase boundaries mathematically.
    No hardcoded day buckets — boundaries emerge from biology.

    Structure:
        Day 1 ... 5             → Menstrual (bleeding)
        Day 6 ... ovulation_day-1 → Follicular (estrogen-dominant)
        Day ovulation_day       → Ovulatory (LH surge, single-day event)
        Day ovulation_day+1 ... cycle_length → Luteal (progesterone-dominant)
    """
    ovulation_day = cycle_length - luteal_length
    # Protect against pathological edge cases
    ovulation_day = max(MENSTRUAL_DURATION + 2, ovulation_day)

    boundaries = {
        "menstrual_end": min(MENSTRUAL_DURATION, ovulation_day - 2),
        "follicular_start": MENSTRUAL_DURATION + 1,
        "follicular_end": ovulation_day - 1,
        "ovulation_day": ovulation_day,
        "luteal_start": ovulation_day + 1,
        "luteal_end": cycle_length,
    }
    return boundaries


def infer_phase(cycle_day, boundaries):
    """
    Node 4: Determine current phase from cycle_day and computed boundaries.
    """
    if cycle_day <= boundaries["menstrual_end"]:
        return "menstrual"
    elif cycle_day <= boundaries["follicular_end"]:
        return "follicular"
    elif cycle_day == boundaries["ovulation_day"]:
        return "ovulatory"
    else:
        return "luteal"


def generate_sleep_quality(phase):
    """
    Node 5: Sleep quality conditioned on phase.
    Causal mechanism: Progesterone fragments sleep architecture (Baker & Driver, 2007).
    Menstrual cramps disrupt sleep onset.
    """
    probs = SLEEP_QUALITY_GIVEN_PHASE[phase]
    categories = list(probs.keys())
    weights = list(probs.values())
    return random.choices(categories, weights=weights, k=1)[0]


def generate_sleep_duration(sleep_quality):
    """
    Node 6: Sleep duration conditioned on sleep quality.
    This is DOWNSTREAM of quality — poor quality causes early waking / fragmentation.
    """
    params = SLEEP_HOURS_GIVEN_QUALITY[sleep_quality]
    raw = np.random.normal(loc=params["mean"], scale=params["sd"])
    clamped = np.clip(raw, params["min"], params["max"])
    return round(float(clamped), 2)


def generate_resting_hr(phase, sleep_quality):
    """
    Node 7: RHR from phase baseline + acute sleep penalty.
    
    Causal chain:
        Phase → hormonal milieu → baseline cardiac sympathovagal balance
        Poor sleep → acute sympathetic activation → additive RHR increase
    
    Grant et al., 2020: Luteal RHR is +2-5 bpm over follicular.
    Zhong et al., 2005: Sleep deprivation → +5-10 bpm acute RHR elevation.
    """
    params = RHR_BASELINE_BY_PHASE[phase]
    baseline = np.random.normal(loc=params["mean"], scale=params["sd"])

    # Causal override: poor sleep induces sympathetic surge
    if sleep_quality == "poor":
        penalty = random.uniform(
            RHR_POOR_SLEEP_PENALTY["min"],
            RHR_POOR_SLEEP_PENALTY["max"]
        )
        baseline += penalty

    clamped = np.clip(baseline, params["min"], params["max"] + 10)  # Allow overshoot from penalty
    return round(float(clamped), 2)


def generate_hrv_trend(phase, sleep_quality):
    """
    Node 8: HRV trend from phase baseline, overridden by sleep.
    
    Causal chain:
        Phase → progesterone/estrogen ratio → vagal tone baseline
        Poor sleep → cortisol spike → overrides vagal recovery
    
    Schmalenberger et al., 2019: HRV lowest in luteal phase.
    """
    if sleep_quality == "poor":
        # Override: acute stress dominates phase-level signal
        probs = HRV_POOR_SLEEP_OVERRIDE
    else:
        probs = HRV_TREND_GIVEN_PHASE[phase]

    categories = list(probs.keys())
    weights = list(probs.values())
    return random.choices(categories, weights=weights, k=1)[0]


def generate_spo2():
    """
    Node 9: SpO2 — physiologically independent of menstrual cycle.
    Tight homeostatic regulation via chemoreceptors.
    No causal link to cycle phase or sleep (in healthy individuals).
    """
    raw = np.random.normal(loc=SPO2_PARAMS["mean"], scale=SPO2_PARAMS["sd"])
    clamped = np.clip(raw, SPO2_PARAMS["min"], SPO2_PARAMS["max"])
    return round(float(clamped), 1)


def generate_temperature_trend(phase):
    """
    Node 10: Temperature trend from phase.
    Shilaih et al., 2018: 0.2–0.5°C rise post-ovulation from progesterone thermogenesis.
    """
    probs = TEMP_TREND_GIVEN_PHASE[phase]
    categories = list(probs.keys())
    weights = list(probs.values())
    return random.choices(categories, weights=weights, k=1)[0]


def generate_symptoms_placeholder(phase, sleep_quality, resting_hr, hrv_trend):
    """
    Node 11: Symptom generation — PLACEHOLDER for LLM integration.
    
    Current implementation uses simple causal rules.
    Future: Feed biometric context to LLM for naturalistic symptom lexicon.
    """
    symptoms = []

    # Causal triggers based on upstream nodes
    is_poor_sleep = sleep_quality == "poor"
    is_elevated_rhr = resting_hr > 70
    is_declining_hrv = hrv_trend in ("decreasing", "erratic")

    # Fatigue: caused by poor sleep OR elevated sympathetic tone
    if is_poor_sleep or is_elevated_rhr:
        if random.random() < 0.75:
            symptoms.append("fatigue")

    # Headache: caused by poor sleep + autonomic instability
    if is_poor_sleep and is_declining_hrv:
        if random.random() < 0.50:
            symptoms.append("headache")

    # Phase-specific symptoms
    if phase == "menstrual":
        if random.random() < 0.70:
            symptoms.append("cramps")
        if random.random() < 0.35:
            symptoms.append("back_pain")
    elif phase == "luteal":
        if random.random() < 0.45:
            symptoms.append("bloating")
        if random.random() < 0.40:
            symptoms.append("breast_tenderness")
        if random.random() < 0.30:
            symptoms.append("mood_changes")
    elif phase == "ovulatory":
        if random.random() < 0.25:
            symptoms.append("bloating")
        if random.random() < 0.15:
            symptoms.append("nausea")

    # Mood changes from autonomic stress
    if is_poor_sleep and is_declining_hrv and "mood_changes" not in symptoms:
        if random.random() < 0.40:
            symptoms.append("mood_changes")

    return symptoms if symptoms else ["none"]


def generate_notes_placeholder():
    """
    Node 12: Notes — PLACEHOLDER for LLM-generated lexical pool.
    Returns a static placeholder. Will be replaced by dynamic generation.
    """
    return "placeholder_note"


# ============================================================
# COMPLETE CLINICAL RECORD GENERATION (PRE-MASKING)
# ============================================================

def generate_complete_clinical_record(sample_id):
    """
    Execute the full causal DAG to produce one complete clinical record.
    Order matters — each node depends only on its causal parents.
    """

    # === LAYER 1: Temporal Structure ===
    cycle_length = generate_cycle_length()
    luteal_length = generate_luteal_phase_length()
    boundaries = infer_phase_boundaries(cycle_length, luteal_length)
    cycle_day = random.randint(1, cycle_length)
    phase = infer_phase(cycle_day, boundaries)

    # === LAYER 2: Sleep (Phase → Quality → Duration) ===
    sleep_quality = generate_sleep_quality(phase)
    sleep_duration_hrs = generate_sleep_duration(sleep_quality)

    # === LAYER 3: Autonomic Biometrics (Phase + Sleep → RHR, HRV) ===
    resting_hr_bpm = generate_resting_hr(phase, sleep_quality)
    hrv_trend = generate_hrv_trend(phase, sleep_quality)

    # === LAYER 4: Independent Physiological (SpO2) ===
    spo2_pct = generate_spo2()

    # === LAYER 5: Temperature (Phase-driven) ===
    temperature_trend = generate_temperature_trend(phase)

    # === LAYER 6: Downstream Symptoms (Biometrics + Phase → Symptoms) ===
    symptoms = generate_symptoms_placeholder(phase, sleep_quality, resting_hr_bpm, hrv_trend)

    # === LAYER 7: User Notes (Placeholder) ===
    notes = generate_notes_placeholder()

    # === Derived metadata ===
    ovulation_day = boundaries["ovulation_day"]
    days_since_ovulation = cycle_day - ovulation_day if cycle_day > ovulation_day else None
    follicular_length = ovulation_day - 1  # Days 1 through ovulation_day-1

    record = {
        # Temporal structure
        "sample_id": sample_id,
        "cycle_length_days": cycle_length,
        "cycle_day": cycle_day,
        "phase": phase,
        "ovulation_day_estimate": ovulation_day,
        "luteal_phase_length_days": luteal_length,
        "follicular_phase_length_days": follicular_length,
        "days_post_ovulation": days_since_ovulation,

        # Sleep (causal chain: phase → quality → duration)
        "sleep_quality": sleep_quality,
        "sleep_duration_hrs": sleep_duration_hrs,

        # Autonomic biometrics (causal chain: phase + sleep → RHR/HRV)
        "resting_hr_bpm": resting_hr_bpm,
        "hrv_trend": hrv_trend,

        # Independent physiological
        "spo2_pct": spo2_pct,

        # Temperature (phase-driven)
        "temperature_trend": temperature_trend,

        # Downstream clinical signals
        "symptoms": symptoms,
        "notes": notes,
    }

    return record


# ============================================================
# USER PROFILE MASKING (SIMULATES REAL-WORLD DATA SPARSITY)
# ============================================================

# Define which fields each profile can observe
# "always" = always present; "probabilistic" = dict of {field: probability}; "never" = always dropped
PROFILE_FIELD_RULES = {
    "minimal_tracker": {
        "always": ["cycle_day", "phase"],
        "probabilistic": {
            "symptoms": 0.60,
            "cycle_length_days": 0.40,
            "notes": 0.30,
        },
        "never": [
            "resting_hr_bpm", "hrv_trend", "spo2_pct",
            "temperature_trend", "sleep_quality", "sleep_duration_hrs",
            "ovulation_day_estimate", "luteal_phase_length_days",
            "follicular_phase_length_days", "days_post_ovulation",
        ],
    },
    "app_logger": {
        "always": ["cycle_day", "phase", "symptoms", "cycle_length_days"],
        "probabilistic": {
            "sleep_quality": 0.70,
            "notes": 0.70,
            "ovulation_day_estimate": 0.50,
        },
        "never": [
            "resting_hr_bpm", "hrv_trend", "spo2_pct",
            "temperature_trend", "sleep_duration_hrs",
            "luteal_phase_length_days", "follicular_phase_length_days",
            "days_post_ovulation",
        ],
    },
    "basic_wearable": {
        "always": [
            "cycle_day", "phase", "resting_hr_bpm",
            "sleep_quality", "sleep_duration_hrs", "spo2_pct",
        ],
        "probabilistic": {
            "hrv_trend": 0.65,
            "symptoms": 0.55,
            "cycle_length_days": 0.60,
            "notes": 0.20,
            "ovulation_day_estimate": 0.30,
        },
        "never": [
            "temperature_trend", "luteal_phase_length_days",
            "follicular_phase_length_days", "days_post_ovulation",
        ],
    },
    "advanced_wearable": {
        "always": [
            "cycle_day", "phase", "resting_hr_bpm", "hrv_trend",
            "temperature_trend", "sleep_quality", "sleep_duration_hrs",
            "spo2_pct", "cycle_length_days", "ovulation_day_estimate",
        ],
        "probabilistic": {
            "symptoms": 0.80,
            "notes": 0.75,
            "luteal_phase_length_days": 0.60,
            "follicular_phase_length_days": 0.60,
            "days_post_ovulation": 0.70,
        },
        "never": [],
    },
    "inconsistent_user": {
        "always": ["cycle_day"],
        "probabilistic": {
            "phase": 0.40,
            "resting_hr_bpm": 0.35,
            "hrv_trend": 0.30,
            "temperature_trend": 0.25,
            "sleep_quality": 0.40,
            "sleep_duration_hrs": 0.35,
            "spo2_pct": 0.30,
            "symptoms": 0.40,
            "cycle_length_days": 0.30,
            "notes": 0.50,
            "ovulation_day_estimate": 0.15,
        },
        "never": [
            "luteal_phase_length_days", "follicular_phase_length_days",
            "days_post_ovulation",
        ],
    },
}

# All maskable fields (clinical data fields, excluding sample_id)
ALL_OBSERVABLE_FIELDS = [
    "cycle_day", "phase", "cycle_length_days", "ovulation_day_estimate",
    "luteal_phase_length_days", "follicular_phase_length_days", "days_post_ovulation",
    "sleep_quality", "sleep_duration_hrs",
    "resting_hr_bpm", "hrv_trend", "spo2_pct",
    "temperature_trend", "symptoms", "notes",
]


def apply_profile_mask(complete_record, profile):
    """
    Apply user profile masking to a complete clinical record.
    Returns the observable sample + metadata about what's available.
    """
    rules = PROFILE_FIELD_RULES[profile]
    always_fields = set(rules["always"])
    probabilistic_fields = rules["probabilistic"]
    never_fields = set(rules["never"])

    observed = {}
    available_fields = []
    missing_fields = []

    for field in ALL_OBSERVABLE_FIELDS:
        if field in never_fields:
            missing_fields.append(field)
            continue

        if field in always_fields:
            if field in complete_record:
                observed[field] = complete_record[field]
                available_fields.append(field)
            else:
                missing_fields.append(field)
            continue

        if field in probabilistic_fields:
            if random.random() < probabilistic_fields[field]:
                if field in complete_record and complete_record[field] is not None:
                    observed[field] = complete_record[field]
                    available_fields.append(field)
                else:
                    missing_fields.append(field)
            else:
                missing_fields.append(field)
            continue

        # Field not mentioned in any rule → treat as unavailable for this profile
        missing_fields.append(field)

    return observed, available_fields, missing_fields


# ============================================================
# CONFIDENCE ESTIMATION (DATA-DRIVEN)
# ============================================================

def estimate_confidence(available_fields, phase):
    """
    Compute a system confidence score based on data completeness and phase clarity.
    More sensor data → higher confidence.
    Some phases are inherently harder to classify.
    """
    # Informative fields for phase prediction
    key_diagnostic_fields = [
        "resting_hr_bpm", "hrv_trend", "temperature_trend",
        "sleep_quality", "sleep_duration_hrs", "spo2_pct", "symptoms",
    ]

    present_count = sum(1 for f in key_diagnostic_fields if f in available_fields)
    completeness_ratio = present_count / len(key_diagnostic_fields)

    # Phase clarity modifier (some phases are more ambiguous)
    phase_clarity = {
        "menstrual": 0.90,  # Bleeding is obvious
        "follicular": 0.70,  # Hard to distinguish mid-cycle
        "ovulatory": 0.75,  # Depends on temp shift detection
        "luteal": 0.80,  # Elevated temp + symptoms help
    }
    clarity = phase_clarity.get(phase, 0.70)

    # Confidence = weighted combination of data coverage and phase signal strength
    base_confidence = 0.30 + (0.50 * completeness_ratio) + (0.20 * clarity)
    # Add small noise
    noise = np.random.normal(0, 0.03)
    confidence = np.clip(base_confidence + noise, 0.15, 0.98)

    return round(float(confidence), 2)


# ============================================================
# MAIN GENERATION PIPELINE
# ============================================================

def build_profile_assignments():
    """
    Create a shuffled list of profile assignments for all N_SAMPLES.
    """
    assignments = []
    for profile, count in PROFILE_DISTRIBUTION.items():
        assignments.extend([profile] * count)

    assert len(assignments) == N_SAMPLES, (
        f"Profile counts sum to {len(assignments)}, expected {N_SAMPLES}"
    )
    random.shuffle(assignments)
    return assignments


def generate_dataset():
    """
    Master pipeline: generate all samples via causal DAG, then mask by profile.
    """
    profile_assignments = build_profile_assignments()
    samples = []
    ground_truths = []

    for i in range(N_SAMPLES):
        sample_id = f"MH-{i + 1:04d}"
        profile = profile_assignments[i]

        # Step 1: Generate complete clinical record via causal chain
        complete_record = generate_complete_clinical_record(sample_id)

        # Step 2: Apply user profile masking
        observed, available_fields, missing_fields = apply_profile_mask(
            complete_record, profile
        )

        # Step 3: Compute system confidence from available data
        phase_for_conf = complete_record["phase"]  # Use true phase for confidence calc
        confidence_score = estimate_confidence(available_fields, phase_for_conf)

        # Step 4: Assemble final sample
        sample = {
            "sample_id": sample_id,
            "data_profile": profile,
            **observed,
            "confidence_score": confidence_score,
            "available_fields": available_fields,
            "missing_fields": missing_fields,
        }

        # Step 5: Store ground truth separately (for LLM evaluation)
        ground_truth = {
            "sample_id": sample_id,
            **complete_record,
        }

        samples.append(sample)
        ground_truths.append(ground_truth)

    return samples, ground_truths


# ============================================================
# VALIDATION & STATISTICS
# ============================================================

def compute_dataset_statistics(samples, ground_truths):
    """Compute summary statistics for the generated dataset."""
    stats = {}

    # Phase distribution from ground truth
    phases = [gt["phase"] for gt in ground_truths]
    phase_counts = {p: phases.count(p) for p in ["menstrual", "follicular", "ovulatory", "luteal"]}
    stats["phase_distribution"] = phase_counts
    stats["phase_percentages"] = {
        p: round(c / len(phases) * 100, 1) for p, c in phase_counts.items()
    }

    # Profile distribution
    profiles = [s["data_profile"] for s in samples]
    stats["profile_distribution"] = {p: profiles.count(p) for p in PROFILE_DISTRIBUTION}

    # Cycle length statistics
    cycle_lengths = [gt["cycle_length_days"] for gt in ground_truths]
    stats["cycle_length"] = {
        "mean": round(np.mean(cycle_lengths), 2),
        "median": round(np.median(cycle_lengths), 2),
        "sd": round(np.std(cycle_lengths), 2),
        "min": int(np.min(cycle_lengths)),
        "max": int(np.max(cycle_lengths)),
        "skewness": round(float(
            3 * (np.mean(cycle_lengths) - np.median(cycle_lengths)) / (np.std(cycle_lengths) + 1e-8)
        ), 3),
    }

    # Luteal phase length statistics
    luteal_lengths = [gt["luteal_phase_length_days"] for gt in ground_truths]
    stats["luteal_phase_length"] = {
        "mean": round(np.mean(luteal_lengths), 2),
        "sd": round(np.std(luteal_lengths), 2),
    }

    # RHR by phase
    rhr_by_phase = {}
    for gt in ground_truths:
        phase = gt["phase"]
        rhr = gt["resting_hr_bpm"]
        if phase not in rhr_by_phase:
            rhr_by_phase[phase] = []
        rhr_by_phase[phase].append(rhr)
    stats["rhr_by_phase"] = {
        p: {"mean": round(np.mean(v), 2), "sd": round(np.std(v), 2)}
        for p, v in rhr_by_phase.items()
    }

    # Sleep quality by phase
    sleep_by_phase = {}
    for gt in ground_truths:
        phase = gt["phase"]
        sq = gt["sleep_quality"]
        if phase not in sleep_by_phase:
            sleep_by_phase[phase] = []
        sleep_by_phase[phase].append(sq)
    stats["sleep_quality_by_phase"] = {
        p: {
            q: round(v.count(q) / len(v) * 100, 1)
            for q in ["poor", "moderate", "good"]
        }
        for p, v in sleep_by_phase.items()
    }

    # Data completeness by profile
    completeness_by_profile = {}
    for s in samples:
        profile = s["data_profile"]
        n_fields = len(s["available_fields"])
        if profile not in completeness_by_profile:
            completeness_by_profile[profile] = []
        completeness_by_profile[profile].append(n_fields)
    stats["mean_available_fields_by_profile"] = {
        p: round(np.mean(v), 2) for p, v in completeness_by_profile.items()
    }

    # Causal coherence check: poor sleep → elevated RHR
    poor_sleep_rhrs = [
        gt["resting_hr_bpm"] for gt in ground_truths if gt["sleep_quality"] == "poor"
    ]
    good_sleep_rhrs = [
        gt["resting_hr_bpm"] for gt in ground_truths if gt["sleep_quality"] == "good"
    ]
    stats["causal_coherence"] = {
        "mean_rhr_poor_sleep": round(np.mean(poor_sleep_rhrs), 2) if poor_sleep_rhrs else None,
        "mean_rhr_good_sleep": round(np.mean(good_sleep_rhrs), 2) if good_sleep_rhrs else None,
        "rhr_delta_poor_vs_good": round(
            np.mean(poor_sleep_rhrs) - np.mean(good_sleep_rhrs), 2
        ) if poor_sleep_rhrs and good_sleep_rhrs else None,
    }

    # HRV override verification
    poor_sleep_hrv = [
        gt["hrv_trend"] for gt in ground_truths if gt["sleep_quality"] == "poor"
    ]
    stats["hrv_when_poor_sleep"] = {
        trend: round(poor_sleep_hrv.count(trend) / len(poor_sleep_hrv) * 100, 1)
        for trend in ["increasing", "stable", "decreasing", "erratic"]
    } if poor_sleep_hrv else {}

    # Confidence by profile
    conf_by_profile = {}
    for s in samples:
        profile = s["data_profile"]
        if profile not in conf_by_profile:
            conf_by_profile[profile] = []
        conf_by_profile[profile].append(s["confidence_score"])
    stats["confidence_by_profile"] = {
        p: round(np.mean(v), 3) for p, v in conf_by_profile.items()
    }

    return stats


# ============================================================
# OUTPUT ASSEMBLY
# ============================================================

def main():
    print("=" * 70)
    print("MENSTRUAL HEALTH DATASET GENERATOR v2.0")
    print("Architecture: Timeline-First Causal DAG")
    print("=" * 70)

    # Generate
    print("\n[1/4] Generating complete clinical records via causal DAG...")
    samples, ground_truths = generate_dataset()
    print(f"      Generated {len(samples)} samples.")

    # Statistics
    print("[2/4] Computing dataset statistics...")
    stats = compute_dataset_statistics(samples, ground_truths)

    # Assemble output
    print("[3/4] Assembling output JSON...")
    output = {
        "metadata": {
            "version": "2.0",
            "architecture": "timeline_first_causal_dag",
            "total_samples": N_SAMPLES,
            "generation_date": datetime.now(timezone.utc).isoformat(),
            "random_seed": SEED,
            "generation_parameters": {
                "cycle_length_distribution": "lognormal",
                "lognormal_mu": LOGNORMAL_MU,
                "lognormal_sigma": LOGNORMAL_SIGMA,
                "cycle_length_bounds": [CYCLE_LENGTH_MIN, CYCLE_LENGTH_MAX],
                "luteal_phase_distribution": "normal",
                "luteal_mean": LUTEAL_MEAN,
                "luteal_sd": LUTEAL_SD,
                "spo2_independent_of_cycle": True,
                "causal_order": [
                    "cycle_length → luteal_length → phase_boundaries → cycle_day → phase",
                    "phase → sleep_quality → sleep_duration",
                    "phase + sleep_quality → resting_hr, hrv_trend",
                    "independent → spo2",
                    "phase → temperature_trend",
                    "phase + sleep + biometrics → symptoms (placeholder)",
                ],
            },
            "profile_distribution": stats["profile_distribution"],
            "phase_distribution": stats["phase_distribution"],
            "phase_percentages": stats["phase_percentages"],
            "cycle_length_statistics": stats["cycle_length"],
            "luteal_phase_statistics": stats["luteal_phase_length"],
            "rhr_by_phase": stats["rhr_by_phase"],
            "sleep_quality_by_phase": stats["sleep_quality_by_phase"],
            "mean_available_fields_by_profile": stats["mean_available_fields_by_profile"],
            "confidence_by_profile": stats["confidence_by_profile"],
            "causal_coherence_validation": stats["causal_coherence"],
            "hrv_override_when_poor_sleep": stats["hrv_when_poor_sleep"],
            "physiological_references": [
                "Bull et al., 2019 — Cycle length lognormal distribution (N=600K+)",
                "Fehring et al., 2006 — Luteal phase consistency (Mean=12.9, SD=1.6)",
                "Shilaih et al., 2018 — BBT biphasic shift post-ovulation",
                "Schmalenberger et al., 2019 — HRV autonomic variation across cycle",
                "Grant et al., 2020 — RHR elevation in luteal phase (+2-5 bpm)",
                "Baker & Driver, 2007 — Sleep architecture disruption in luteal",
                "Zhong et al., 2005 — Acute sleep deprivation → sympathetic RHR surge",
                "Kräuchi, 2007 — Thermoregulatory cascade and sleep onset",
            ],
            "notes": {
                "symptoms": "PLACEHOLDER — to be replaced by LLM-generated lexical pool",
                "user_notes": "PLACEHOLDER — to be replaced by dynamic LLM generation",
            },
        },
        "samples": samples,
        "ground_truth": ground_truths,
    }

    # Save
    output_path = "data/structured/menstrual_health_dataset.json"
    print(f"[4/4] Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 70}")

    # Print summary
    print(f"\n{'─' * 70}")
    print("DATASET STATISTICS")
    print(f"{'─' * 70}")

    print(f"\n  Cycle Length Distribution (Lognormal):")
    cl = stats["cycle_length"]
    print(f"    Mean: {cl['mean']} days | Median: {cl['median']} days | SD: {cl['sd']}")
    print(f"    Range: [{cl['min']}, {cl['max']}] | Skewness: {cl['skewness']}")

    print(f"\n  Luteal Phase: Mean = {stats['luteal_phase_length']['mean']}, "
          f"SD = {stats['luteal_phase_length']['sd']}")

    print(f"\n  Phase Distribution:")
    for phase, pct in stats["phase_percentages"].items():
        count = stats["phase_distribution"][phase]
        print(f"    {phase:12s}: {count:4d} ({pct}%)")

    print(f"\n  Resting HR by Phase (mean ± sd):")
    for phase, vals in stats["rhr_by_phase"].items():
        print(f"    {phase:12s}: {vals['mean']} ± {vals['sd']} bpm")

    print(f"\n  Causal Coherence Validation (Sleep → RHR):")
    cc = stats["causal_coherence"]
    print(f"    Mean RHR (poor sleep):  {cc['mean_rhr_poor_sleep']} bpm")
    print(f"    Mean RHR (good sleep):  {cc['mean_rhr_good_sleep']} bpm")
    print(f"    Delta (poor - good):    +{cc['rhr_delta_poor_vs_good']} bpm ✓")

    print(f"\n  HRV Trend When Sleep is Poor (override validation):")
    for trend, pct in stats["hrv_when_poor_sleep"].items():
        marker = "✓" if trend in ("decreasing", "erratic") else ""
        print(f"    {trend:12s}: {pct}% {marker}")

    print(f"\n  Mean Available Fields by Profile:")
    for profile, mean_f in stats["mean_available_fields_by_profile"].items():
        print(f"    {profile:20s}: {mean_f}")

    print(f"\n  Confidence Score by Profile:")
    for profile, conf in stats["confidence_by_profile"].items():
        print(f"    {profile:20s}: {conf}")

    print(f"\n  Sleep Quality by Phase (%):")
    for phase, quals in stats["sleep_quality_by_phase"].items():
        print(f"    {phase:12s}: poor={quals['poor']}% | "
              f"moderate={quals['moderate']}% | good={quals['good']}%")

    print(f"\n{'─' * 70}")
    print(f"  Output: {output_path}")
    print(f"  Samples: {N_SAMPLES} (observed) + {N_SAMPLES} (ground truth)")
    print(f"  Seed: {SEED}")
    print(f"{'─' * 70}\n")


if __name__ == "__main__":
    main()