# Analysis 4b: Survival Analysis

## Key Findings

- **This analysis is hypothesis-generating.** With n=35 discordant
  patients and 5 events, we are severely underpowered
  to detect clinically meaningful survival differences.
- Hazard ratio (discordant vs concordant negative): HR=1.20 (95% CI: 0.49-2.98, p=0.6906).
- The wide confidence interval reflects severe underpowering.
- Log-rank test: chi2=0.16, p=0.6883

## Methods

Kaplan-Meier survival curves were generated for Overall Survival (OS) comparing
Concordant Negative, Discordant (IHC-/RNA-high), and Concordant Positive groups.
Log-rank test compared Concordant Negative vs Discordant. A univariate Cox
proportional hazards model estimated the hazard ratio for discordant status.

Survival status was parsed from TCGA encoding (1:DECEASED = event, 0:LIVING = censored).

## Results

### Overall Survival

| Group | N | Events | Median OS (months) |
|---|---|---|---|
| Concordant Negative | 651 | 73 | 122.7 |
| Discordant | 35 | 5 | Not reached |
| Concordant Positive | 151 | 17 | 216.6 |

**Log-rank test (Concordant Neg vs Discordant):** chi2=0.16, p=0.6883

**Cox Proportional Hazards (univariate):**
- HR (discordant vs concordant negative) = 1.20 (95% CI: 0.49-2.98)
- p = 0.6906

### Disease-Free Survival

| Group | N | Events | Median DFS (months) |
|---|---|---|---|
| Concordant Negative | 608 | 62 | Not reached |
| Discordant | 31 | 2 | Not reached |
| Concordant Positive | 137 | 11 | 214.7 |

## Limitations

- **Severely underpowered.** The discordant group has n=35 patients with
  5 events. Standard power calculations suggest n>100 per arm
  for detecting HR differences of 1.5 with 80% power. Our analysis has <10% of this
  requirement.
- TCGA follow-up is variable; censoring patterns may differ across institutions.
- Treatment data is not available; survival differences (or lack thereof) cannot be
  attributed to HER2 status vs treatment vs confounders.
- No adjustment for age, stage, ER status, or other prognostic factors (insufficient
  sample for multivariable modeling in the discordant group).

## Implications

The point estimate (HR=1.20) is near null, suggesting no large survival
difference between discordant and concordant negative patients. This does not rule out
a clinically meaningful effect -- the study is severely underpowered.

---

**Figures:**
- `fig_04_4b_km_os.png` -- Kaplan-Meier curves for OS
- `fig_04_4b_km_dfs.png` -- Kaplan-Meier curves for DFS
