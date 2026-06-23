# Biomarker Actionability

This codebase is built as a portfolio project demonstrating decision curve analysis applied to clinical biomarker evaluation.

It describes a decision-focused evaluation of when biological and digital biomarkers actually change clinical decisions rather than just whether they improve prediction.

## The question

Prediction and actionability are different problems. A model can be more accurate without changing what a clinician would do. This project tests whether adding wearable activity data to standard blood biomarkers changes a cardiovascular screening decision, using decision curve analysis (DCA) rather than accuracy metrics alone.

DCA asks: at a given risk threshold, does using a model to guide treatment actually help more patients than treating everyone or no one? A higher area under the ROC curve (AUC) does not guarantee a higher net benefit. This distinction matters because the case for deploying wearables in clinical practice depends on whether they change decisions, not just on whether they improve a ranking metric.

## Pipeline

| Notebook | What it does |
|----------|-------------|
| `01_data_assembly` | Merge NHANES (National Health and Nutrition Examination Survey) 2011-2014 XPT tables, filter to fasting subsample, rename columns, aggregate accelerometer daily data |
| `02_feature_engineering` | Define hard cardiovascular disease (CVD) outcome (self-reported myocardial infarction (MI), stroke or coronary heart disease (CHD)), apply standard accelerometer validity criteria, build biological/digital/combined feature sets |
| `03_modelling` | LightGBM (Optuna-tuned, Brier score objective) and TabPFN across three feature sets; 5-fold stratified CV with fold-wise median imputation; out-of-fold and held-out test predictions; calibration curves |
| `04_decision_curve_analysis` | Net benefit across 1-30% thresholds; age-alone baseline via logistic regression; bootstrap 95% CIs; OOF vs test consistency check |
| `05_subgroup_analysis` | Decision curves by age band, sex and BMI category using out-of-fold (OOF) predictions |

## Key findings

The dataset covers 3,868 participants (OOF set), 290 hard CVD cases with 7.5% prevalence.

Age alone achieves AUC 0.845. The full 19-feature blood panel achieves approximately 0.852, a gap of just 0.007.

Adding wearable-derived activity features (daily MIMS (Monitor-Independent Movement Summary, the accelerometer activity metric designed to be comparable across device brands) sum and variability) to the blood panel did not improve net benefit at any clinically relevant threshold. The `find_actionable_range` function returned no actionable range at population level. TabPFN was slightly better calibrated than LightGBM in the screening threshold range but this did not shift the decision curves.

Subgroup analysis found marginal actionable ranges in adults aged 60-79 (thresholds 0.125-0.260) and the overweight BMI group (0.125-0.305). All other subgroups showed no actionable range. The 80+ age band had no participants, reflecting physical activity monitor (PAM) study design exclusions.

A sensitivity analysis suggests a wearable signal would need to shift risk by roughly 5 percentage points in patients before it changed screening decisions, which is likely beyond what a one-week cross-sectional snapshot can provide.

## Interpretation

Cross-sectional wearable data adds no decision value beyond blood markers in this dataset. Whatever a one week long activity snapshot captures about cardiovascular risk is already captured by age and the metabolic panel. This is likely a limitation of the measurement design, not of wearables as a concept. The value of wearables likely lies in longitudinal monitoring (tracking behavioural change over months) which this dataset cannot test.

This result does not support the hypothesis that adding a wearable activity metric to a standard blood panel improves clinical decision making for CVD screening in this data set. 

## Limitations

- Cross-sectional design: models predict prevalent CVD, not incident events. Cannot capture risk trajectories or deterioration.
- Self-reported CVD outcomes (survey question), not events from medical records.
- Small case count (290 in the OOF set) limits statistical power. Confidence intervals are wide and subgroup analyses are underpowered; only the 60-79 and overweight subgroups have sufficient cases for reliable DCA.
- Seven-day accelerometry from 2011-2014 using MIMS. Not equivalent to modern continuous wearables or heart-rate based metrics.
- hs-CRP and other inflammatory markers are not available in these NHANES cycles, leaving a gap in the biological feature set.
- Single dataset with no external validation.
- LightGBM final test models use hyperparameters from the last CV fold rather than a separate tuning set. This is a documented compromise that slightly inflates variance in the reported test metrics.

## Reproducibility

Raw NHANES XPT files are not tracked (see `.gitignore`). Download them from the CDC NHANES website (2011-2012 and 2013-2014 cycles) and place under `data/raw/2011-2012/` and `data/raw/2013-2014/`.

TabPFN requires an API token. Set it via the environment variable `TABPFN_TOKEN` before running notebook 03. A `.env` file in the project root (already gitignored) is the recommended approach (do not hardcode tokens in notebooks)

```
# .env
TABPFN_TOKEN=your_token_here
```

Run notebooks in order (01 through 05). Each saves intermediate results to `data/processed/` as pickle files consumed by the next notebook.
