import pandas as pd

# ── Feature set definitions ───────────────────────────────────────────────────
# Single source of truth — imported by all downstream notebooks and scripts.

BIOLOGICAL_FEATURES = [
    # Demographics
    'age_years', 'sex', 'race_ethnicity', 'family_poverty_income_ratio',
    # Lipids
    'total_cholesterol_mg_dl', 'hdl_mg_dl', 'ldl_mg_dl', 'triglycerides_mg_dl',
    'chol_hdl_ratio', 'non_hdl_cholesterol',
    # Glycaemia (HbA1c preferred; glucose retained for completeness)
    'hba1c_pct', 'fasting_glucose_mg_dl',
    # Blood pressure
    'systolic_bp_1', 'diastolic_bp_1', 'pulse_pressure',
    # Adiposity (waist preferred over BMI)
    'waist_circumference_cm', 'bmi',
    # Lifestyle
    'ever_smoker', 'current_smoker',
]

DIGITAL_FEATURES = [
    'mims_mean',           # average daily total activity
    'mims_sd',             # day-to-day variability in activity
    'mims_daily_cv',       # coefficient of variation (normalised variability)
    'inactive_min_mean',   # average daily sedentary time
    'vigorous_min_mean',   # average daily vigorous activity
    'wake_wear_min_mean',  # average daily wear time (compliance proxy)
    'nonwear_min_mean',    # average daily non-wear time
    'valid_days',          # number of valid days (data quality indicator)
]

COMBINED_FEATURES = BIOLOGICAL_FEATURES + DIGITAL_FEATURES

TARGETS = ['cvd_hard', 'cvd_composite']


# ── Feature engineering functions ─────────────────────────────────────────────

def aggregate_accelerometer(
    df: pd.DataFrame,
    min_valid_days: int = 4,
    min_wake_wear_min: int = 600,
) -> pd.DataFrame:
    """
    Quality-filter accelerometer days and aggregate to one row per participant.

    Valid day criteria (Troiano et al. Med Sci Sports Exerc 2008):
      - accel_quality_flag == 0 (good quality)
      - accel_wake_wear_min >= min_wake_wear_min (default 600 = 10 hours)

    Parameters
    ----------
    df : DataFrame with one row per accelerometer day per participant.
    min_valid_days : minimum valid days required to include a participant.
    min_wake_wear_min : minimum wake wear minutes for a day to count as valid.

    Returns
    -------
    DataFrame with one row per participant, columns:
        participant_id, valid_days, mims_mean, mims_sd, mims_daily_cv,
        inactive_min_mean, vigorous_min_mean, wake_wear_min_mean,
        nonwear_min_mean
    """
    accel_raw = df[df['accel_mims_sum_daily'].notna()].copy()

    valid_days = accel_raw[
        (accel_raw['accel_quality_flag'] == 0) &
        (accel_raw['accel_wake_wear_min'] >= min_wake_wear_min)
    ]

    agg = valid_days.groupby('participant_id').agg(
        valid_days         =('accel_mims_sum_daily',     'count'),
        mims_mean          =('accel_mims_sum_daily',     'mean'),
        mims_sd            =('accel_mims_sum_daily',     'std'),
        inactive_min_mean  =('accel_awake_inactive_min', 'mean'),
        vigorous_min_mean  =('accel_vigorous_min',       'mean'),
        wake_wear_min_mean =('accel_wake_wear_min',      'mean'),
        nonwear_min_mean   =('accel_nonwear_min',        'mean'),
        mims_daily_cv      =('accel_mims_sum_daily',     lambda x: x.std() / x.mean()),
    ).reset_index()

    agg = agg[agg['valid_days'] >= min_valid_days].reset_index(drop=True)
    return agg


def build_biological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and derive biological features at participant level.

    Takes the first row per participant (biomarker data is constant across
    accelerometer days) and engineers derived clinical features.

    Derived features
    ----------------
    chol_hdl_ratio      : Total cholesterol / HDL — used in Framingham risk score
    non_hdl_cholesterol : Total cholesterol - HDL — captures all atherogenic lipoproteins
    pulse_pressure      : Systolic BP - Diastolic BP — marker of arterial stiffness
    ever_smoker         : smoked_100_cigarettes_lifetime == 1
    current_smoker      : currently_smoke_cigarettes == 1

    Parameters
    ----------
    df : DataFrame with one or more rows per participant.

    Returns
    -------
    DataFrame with one row per participant containing biological features.
    """
    source_cols = [
        'age_years', 'sex', 'race_ethnicity', 'family_poverty_income_ratio',
        'total_cholesterol_mg_dl', 'hdl_mg_dl', 'ldl_mg_dl', 'triglycerides_mg_dl',
        'hba1c_pct', 'fasting_glucose_mg_dl',
        'systolic_bp_1', 'diastolic_bp_1',
        'bmi', 'waist_circumference_cm',
        'smoked_100_cigarettes_lifetime', 'currently_smoke_cigarettes',
    ]

    bio = (
        df.groupby('participant_id')[[c for c in source_cols if c in df.columns]]
        .first()
        .reset_index()
    )

    bio['chol_hdl_ratio']      = bio['total_cholesterol_mg_dl'] / bio['hdl_mg_dl']
    bio['non_hdl_cholesterol'] = bio['total_cholesterol_mg_dl'] - bio['hdl_mg_dl']
    bio['pulse_pressure']      = bio['systolic_bp_1'] - bio['diastolic_bp_1']

    # NHANES smoking coding: 1 = Yes, 2 = No
    if 'smoked_100_cigarettes_lifetime' in bio.columns:
        bio['ever_smoker'] = (bio['smoked_100_cigarettes_lifetime'] == 1).astype(int)
        bio = bio.drop(columns=['smoked_100_cigarettes_lifetime'])
    if 'currently_smoke_cigarettes' in bio.columns:
        bio['current_smoker'] = (bio['currently_smoke_cigarettes'] == 1).astype(int)
        bio = bio.drop(columns=['currently_smoke_cigarettes'])

    return bio
