import json
import pandas as pd
 
RENAME_MAP = {
    # Identifiers
    'SEQN': 'participant_id',
 
    # Body measures
    'BMDSTATS': 'body_measures_status',
    'BMXWT': 'weight_kg',
    'BMIWT': 'weight_comment',
    'BMXRECUM': 'recumbent_length_cm',
    'BMIRECUM': 'recumbent_length_comment',
    'BMXHEAD': 'head_circumference_cm',
    'BMIHEAD': 'head_circumference_comment',
    'BMXHT': 'height_cm',
    'BMIHT': 'height_comment',
    'BMXBMI': 'bmi',
    'BMDBMIC': 'bmi_category_children',
    'BMXLEG': 'upper_leg_length_cm',
    'BMILEG': 'upper_leg_length_comment',
    'BMXARML': 'arm_length_cm',
    'BMIARML': 'arm_length_comment',
    'BMXARMC': 'arm_circumference_cm',
    'BMIARMC': 'arm_circumference_comment',
    'BMXWAIST': 'waist_circumference_cm',
    'BMIWAIST': 'waist_circumference_comment',
    'BMXSAD1': 'sagittal_abdominal_diameter_1',
    'BMXSAD2': 'sagittal_abdominal_diameter_2',
    'BMXSAD3': 'sagittal_abdominal_diameter_3',
    'BMXSAD4': 'sagittal_abdominal_diameter_4',
    'BMDAVSAD': 'sagittal_abdominal_diameter_avg',
    'BMDSADCM': 'sagittal_abdominal_diameter_comment',
 
    # Blood pressure questionnaire
    'BPQ020': 'ever_told_high_bp',
    'BPQ030': 'told_high_bp_2plus_times',
    'BPD035': 'age_first_told_high_bp',
    'BPQ040A': 'taking_bp_medication',
    'BPQ050A': 'currently_taking_bp_medication',
    'BPQ056': 'days_since_last_bp_medication',
    'BPD058': 'age_started_bp_medication',
    'BPQ059': 'ever_told_low_bp',
    'BPQ080': 'ever_told_high_cholesterol',
    'BPQ060': 'how_long_ago_cholesterol_checked',
    'BPQ070': 'taking_cholesterol_medication',
    'BPQ090D': 'told_to_control_weight_for_chol',
    'BPQ100D': 'now_controlling_weight_for_chol',
 
    # Blood pressure exam
    'PEASCST1': 'bp_exam_status',
    'PEASCTM1': 'bp_exam_comment',
    'PEASCCT1': 'bp_exam_component_status',
    'BPXCHR': 'pulse_60sec_count',
    'BPAARM': 'bp_arm_used',
    'BPACSZ': 'bp_cuff_size',
    'BPXPLS': 'pulse_rate',
    'BPXPULS': 'pulse_regularity',
    'BPXPTY': 'pulse_type',
    'BPXML1': 'max_inflation_level_mmhg',
    'BPXSY1': 'systolic_bp_1',
    'BPXDI1': 'diastolic_bp_1',
    'BPAEN1': 'bp_reading_1_comment',
    'BPXSY2': 'systolic_bp_2',
    'BPXDI2': 'diastolic_bp_2',
    'BPAEN2': 'bp_reading_2_comment',
    'BPXSY3': 'systolic_bp_3',
    'BPXDI3': 'diastolic_bp_3',
    'BPAEN3': 'bp_reading_3_comment',
    'BPXSY4': 'systolic_bp_4',
    'BPXDI4': 'diastolic_bp_4',
    'BPAEN4': 'bp_reading_4_comment',
 
    # Demographics
    'SDDSRVYR': 'survey_cycle',
    'RIDSTATR': 'interview_exam_status',
    'RIAGENDR': 'sex',
    'RIDAGEYR': 'age_years',
    'RIDAGEMN': 'age_months',
    'RIDRETH1': 'race_ethnicity',
    'RIDRETH3': 'race_ethnicity_with_asian',
    'RIDEXMON': 'exam_month_nov_to_apr',
    'RIDEXAGM': 'age_months_at_exam',
    'DMQMILIZ': 'served_active_duty_military',
    'DMQADFC': 'served_in_foreign_conflict',
    'DMDBORN4': 'country_of_birth',
    'DMDCITZN': 'citizenship_status',
    'DMDYRSUS': 'years_in_us',
    'DMDEDUC3': 'education_child',
    'DMDEDUC2': 'education_adult',
    'DMDMARTL': 'marital_status',
    'RIDEXPRG': 'pregnancy_status',
 
    # Language and proxy
    'SIALANG': 'screener_interview_language',
    'SIAPROXY': 'screener_proxy_used',
    'SIAINTRP': 'screener_interpreter_used',
    'FIALANG': 'family_interview_language',
    'FIAPROXY': 'family_proxy_used',
    'FIAINTRP': 'family_interpreter_used',
    'MIALANG': 'mec_interview_language',
    'MIAPROXY': 'mec_proxy_used',
    'MIAINTRP': 'mec_interpreter_used',
    'AIALANGA': 'acasi_language',
 
    # Survey weights and design
    'WTINT2YR': 'weight_interview_2yr',
    'WTMEC2YR': 'weight_mec_2yr',
    'SDMVPSU': 'psu_masked_variance',
    'SDMVSTRA': 'stratum_masked_variance',
 
    # Income
    'INDHHIN2': 'household_income_range',
    'INDFMIN2': 'family_income_range',
    'INDFMPIR': 'family_poverty_income_ratio',
 
    # Household
    'DMDHHSIZ': 'household_size',
    'DMDFMSIZ': 'family_size',
    'DMDHHSZA': 'num_children_5_and_under',
    'DMDHHSZB': 'num_children_6_to_17',
    'DMDHHSZE': 'num_adults_60_and_over',
    'DMDHRGND': 'household_ref_person_sex',
    'DMDHRAGE': 'household_ref_person_age',
    'DMDHRBR4': 'household_ref_person_birth_country',
    'DMDHREDU': 'household_ref_person_education',
    'DMDHRMAR': 'household_ref_person_marital_status',
    'DMDHSEDU': 'household_ref_spouse_education',
 
    # Laboratory - HbA1c
    'LBXGH': 'hba1c_pct',
 
    # Laboratory - Glucose (fasting)
    'WTSAF2YR_x': 'weight_fasting_2yr_glucose',
    'LBXGLU': 'fasting_glucose_mg_dl',
    'LBDGLUSI': 'fasting_glucose_mmol_l',
    'PHAFSTHR': 'fasting_hours',
    'PHAFSTMN': 'fasting_minutes',
 
    # Laboratory - HDL
    'LBXHDD': 'hdl_mg_dl',
    'LBDHDD': 'hdl_mg_dl',
    'LBDHDDSI': 'hdl_mmol_l',
 
    # Laboratory - Total cholesterol
    'LBXTC': 'total_cholesterol_mg_dl',
    'LBDTCSI': 'total_cholesterol_mmol_l',
 
    # Laboratory - Triglycerides and LDL
    'WTSAF2YR_y': 'weight_fasting_2yr_lipids',
    'LBXTR': 'triglycerides_mg_dl',
    'LBDTRSI': 'triglycerides_mmol_l',
    'LBDLDL': 'ldl_mg_dl',
    'LBDLDLSI': 'ldl_mmol_l',
 
    # Medical conditions questionnaire
    'MCQ010': 'ever_told_asthma',
    'MCQ025': 'age_first_asthma',
    'MCQ035': 'still_have_asthma',
    'MCQ040': 'asthma_attack_past_year',
    'MCQ050': 'er_visit_asthma_past_year',
    'MCQ053': 'taking_asthma_medication',
    'MCQ070': 'ever_told_psoriasis',
    'MCQ075': 'ever_told_celiac_disease',
    'MCQ080': 'doctor_said_overweight',
    'MCQ082': 'ever_told_celiac_disease_2',
    'MCQ084': 'ever_told_memory_loss',
    'MCQ086': 'memory_loss_getting_worse',
    'MCQ092': 'ever_told_blood_transfusion',
    'MCD093': 'year_of_blood_transfusion',
    'MCQ149': 'menstrual_periods_started',
    'MCQ160A': 'ever_told_arthritis',
    'MCQ180A': 'age_told_arthritis',
    'MCQ195': 'arthritis_type',
    'MCQ160N': 'ever_told_gout',
    'MCQ180N': 'age_told_gout',
    'MCQ160B': 'ever_told_congestive_heart_failure',
    'MCQ180B': 'age_told_chf',
    'MCQ160C': 'ever_told_coronary_heart_disease',
    'MCQ180C': 'age_told_chd',
    'MCQ160D': 'ever_told_angina',
    'MCQ180D': 'age_told_angina',
    'MCQ160E': 'ever_told_heart_attack',
    'MCQ180E': 'age_told_heart_attack',
    'MCQ160F': 'ever_told_stroke',
    'MCQ180F': 'age_told_stroke',
    'MCQ160G': 'ever_told_emphysema',
    'MCQ180G': 'age_told_emphysema',
    'MCQ160M': 'ever_told_thyroid_problem',
    'MCQ170M': 'still_have_thyroid_problem',
    'MCQ180M': 'age_told_thyroid',
    'MCQ160K': 'ever_told_chronic_bronchitis',
    'MCQ170K': 'still_have_chronic_bronchitis',
    'MCQ180K': 'age_told_chronic_bronchitis',
    'MCQ160L': 'ever_told_liver_condition',
    'MCQ170L': 'still_have_liver_condition',
    'MCQ180L': 'age_told_liver_condition',
    'MCQ220': 'ever_told_cancer',
    'MCQ230A': 'first_cancer_type_a',
    'MCQ230B': 'first_cancer_type_b',
    'MCQ230C': 'first_cancer_type_c',
    'MCQ230D': 'first_cancer_type_d',
    'MCQ240A': 'age_bladder_cancer',
    'MCQ240AA': 'age_other_cancer',
    'MCQ240B': 'age_blood_cancer',
    'MCQ240BB': 'age_other_cancer_2',
    'MCQ240C': 'age_bone_cancer',
    'MCQ240CC': 'age_other_cancer_3',
    'MCQ240D': 'age_brain_cancer',
    'MCQ240DD': 'age_other_cancer_4',
    'MCQ240DK': 'age_cancer_dont_know',
    'MCQ240E': 'age_breast_cancer',
    'MCQ240F': 'age_cervical_cancer',
    'MCQ240G': 'age_colon_cancer',
    'MCQ240H': 'age_esophageal_cancer',
    'MCQ240I': 'age_gallbladder_cancer',
    'MCQ240J': 'age_kidney_cancer',
    'MCQ240K': 'age_larynx_cancer',
    'MCQ240L': 'age_leukemia',
    'MCQ240M': 'age_liver_cancer',
    'MCQ240N': 'age_lung_cancer',
    'MCQ240O': 'age_lymphoma',
    'MCQ240P': 'age_melanoma',
    'MCQ240Q': 'age_mouth_cancer',
    'MCQ240R': 'age_nervous_system_cancer',
    'MCQ240S': 'age_ovarian_cancer',
    'MCQ240T': 'age_pancreatic_cancer',
    'MCQ240U': 'age_prostate_cancer',
    'MCQ240V': 'age_rectal_cancer',
    'MCQ240W': 'age_skin_nonmelanoma_cancer',
    'MCQ240X': 'age_skin_unknown_cancer',
    'MCQ240Y': 'age_soft_tissue_cancer',
    'MCQ240Z': 'age_stomach_cancer',
    'MCQ300A': 'family_history_heart_attack',
    'MCQ300B': 'family_history_asthma',
    'MCQ300C': 'family_history_diabetes',
    'MCQ365A': 'doctor_told_lose_weight',
    'MCQ365B': 'doctor_told_exercise_more',
    'MCQ365C': 'doctor_told_reduce_salt',
    'MCQ365D': 'doctor_told_reduce_fat',
    'MCQ370A': 'currently_controlling_weight',
    'MCQ370B': 'currently_exercising_more',
    'MCQ370C': 'currently_reducing_salt',
    'MCQ370D': 'currently_reducing_fat',
    'MCQ380': 'healthy_weight_self_perception',
 
    # Accelerometer (daily summary)
    'PAXDAYD': 'accel_day_of_data',
    'PAXDAYWD': 'accel_weekday_weekend',
    'PAXSSNDP': 'accel_data_start_second',
    'PAXMSTD': 'accel_mims_sum_daily',
    'PAXTMD': 'accel_total_minutes_daily',
    'PAXAISMD': 'accel_awake_inactive_min',
    'PAXVMD': 'accel_vigorous_min',
    'PAXMTSD': 'accel_mims_sd_daily',
    'PAXWWMD': 'accel_wake_wear_min',
    'PAXSWMD': 'accel_sleep_wear_min',
    'PAXNWMD': 'accel_nonwear_min',
    'PAXUMD': 'accel_unknown_min',
    'PAXLXSD': 'accel_log_mims_sd',
    'PAXQFD': 'accel_quality_flag',
 
    # Prescriptions
    'RXDUSE': 'takes_prescription_medication',
    'RXDDRUG': 'prescription_drug_name',
    'RXDDRGID': 'prescription_drug_id',
    'RXQSEEN': 'prescription_container_seen',
    'RXDDAYS': 'prescription_days_taken',
    'RXDCOUNT': 'num_prescription_drugs',
 
    # Smoking
    'SMQ020': 'smoked_100_cigarettes_lifetime',
    'SMD030': 'age_started_smoking',
    'SMQ040': 'currently_smoke_cigarettes',
    'SMQ050Q': 'how_long_since_quit_smoking',
    'SMQ050U': 'quit_smoking_time_unit',
    'SMD055': 'age_last_smoked_regularly',
    'SMD057': 'cigarettes_per_day_when_smoked',
    'SMD641': 'cigarettes_per_day_past_5days',
    'SMD650': 'cigarettes_per_day_usual',
    'SMD093': 'may_i_see_cigarette_pack',
    'SMDUPCA': 'cigarette_upc_code',
    'SMD100BR': 'cigarette_brand',
    'SMD100FL': 'cigarette_filtered',
    'SMD100MN': 'cigarette_menthol',
    'SMD100LN': 'cigarette_length',
    'SMD100TR': 'cigarette_tar_mg',
    'SMD100NI': 'cigarette_nicotine_mg',
    'SMD100CO': 'cigarette_co_mg',
    'SMQ621': 'used_tobacco_products_past_5days',
    'SMD630': 'age_first_tobacco_product',
    'SMQ670': 'tried_to_quit_smoking_past_year',
    'SMAQUEX2': 'smoking_questionnaire_source',
}
 
 
def rename_columns(df, save_path='data/raw/column_rename_map.json'):
    """
    Rename NHANES coded columns to human-readable names.
    Saves the mapping to a JSON file for reference.
    Returns the renamed DataFrame.
    """
    # Save the full mapping (only keys that exist in the dataframe)
    used_map = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    unmapped = [c for c in df.columns if c not in RENAME_MAP]
 
    reference = {
        'renamed': used_map,
        'unmapped_columns': unmapped,
    }
 
    with open(save_path, 'w') as f:
        json.dump(reference, f, indent=2)
 
    print(f"Renamed {len(used_map)} columns. {len(unmapped)} columns unmapped.")
    if unmapped:
        print(f"Unmapped: {unmapped}")
 
    return df.rename(columns=used_map)