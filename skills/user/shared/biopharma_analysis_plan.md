# Biopharma Deliverable Analysis Plan

**Date:** 2026-04-06
**Status:** High-level plan, pre-implementation

---

## Strategic Context

The primary opportunity for biopharma decision-making is twofold:

1. **Discordant group (newly identified population):** HER2-NEG/ERBB2-HIGH patients
   split into CN>=2 (likely FN by IHC) and CN<=1 (potentially ER-driven biology,
   candidates for hormone modulatory treatment). Long-term objective: companion
   diagnostic filing and/or retroactive cohort study toward label expansion.

2. **Equivocal group (existing clinical pain point):** IHC 2+ patients requiring
   FISH reflex testing. RNA-based classification could replace or augment FISH,
   with a cleaner and more immediate regulatory pathway.

The equivocal story is the "land" play (lower risk, faster path, established clinical
need). The discordant story is the "expand" play (larger but longer-term opportunity).
A biopharma pitch leads with equivocal, then pivots to discordant.

| Dimension | Equivocal Story | Discordant Story |
|---|---|---|
| Clinical need | Already recognized (FISH reflex) | Newly identified (missed patients) |
| CDx pathway | Predicate comparison (RNA vs. FISH) | De novo CDx (new biomarker) |
| Regulatory complexity | Lower (replacing existing test) | Higher (defining new indication) |
| Timeline to value | Near-term | Medium-to-long-term |
| Biopharma action | File CDx for existing HER2 drugs | Retroactive cohort study -> label expansion |
| Data requirement | Moderate (concordance study) | High (RWD + prospective study) |

**Note:** Actual analyses to meet these objectives require Tempus real-world data
(not available here). The analyses below use TCGA data to build the prerequisite
evidence and demonstrate analytical capability.

---

## Proposed Analyses

### Analysis #1: HER2 Testing Method as Confounder

**Question:** Was the testing method used a confounder to HER2 negative readout
among discordant patients?

**Data:** `her2_test_method` (cleaned) / `HER2 positivity method text` (original),
`HER2 fish method` in clinical data.

**Feasibility concern:** TCGA clinical annotations for testing methodology are
notoriously sparse. Before committing, count how many discordant patients have
this field populated. If <10, the analysis degrades to a descriptive table with
a caveat, not a true confounder analysis. Still worth doing as due diligence.

**Priority:** 6 of 9 (due diligence, likely data-limited)

---

### Analysis #2: Prevalence of Molecular ERBB2 Overexpression in IHC-Negative

**Question:** What is the approximate size of the "missed HER2+" population?

**Framing correction:** Do not call this a "false negative rate" -- IHC/FISH is the
clinical ground truth, so there is no higher authority to define FN against. The
defensible framing is: **"prevalence of molecular ERBB2 overexpression among
IHC-negative patients."** Then the argument becomes: "X% of IHC-negative patients
show molecular evidence that questions their negative classification."

- "We estimate a FN rate of X%" invites immediate challenge on truth standard.
- "We identify a molecular subgroup comprising X% of the IHC-negative population
  with ERBB2 transcriptional profiles indistinguishable from IHC-positive patients"
  is more robust and harder to attack.

Method-specific breakdown from #1 is exploratory stratification of this prevalence,
not a method-specific error rate (unless N supports it).

**Priority:** 3 of 9 (essential for population sizing)

---

### Analysis #3: Discordant Biology -- Normal Tissue and ER Pathway

**Question:** Is the non-amplified discordant group (CN<=1) driven by unusual
baseline biology or disease-driven ER/luminal co-regulation?

**Approach:**
- Compare tumor-to-matched-normal ERBB2 ratios (not just absolute tumor levels)
  for discordant vs. concordant-negative patients. Controls for individual variation.
- Explicit correlation of ERBB2 with FOXA1, ESR1, and select ER pathway genes
  within the non-amplified discordant group.
- Use the cleaned ER quantitative subscales (`er_allred_score`, `er_hscore`,
  `er_intensity`, `er_percent_positive`) -- if the non-amplified discordant group
  is truly ER-driven, expect elevated ER quantitative scores, not just binary ER+.

**Interpretation framework:**
- High ERBB2 + high FOXA1 + high ESR1 = luminal biology with incidental ERBB2
  co-regulation -> favors endocrine therapy
- High ERBB2 + low ESR1 + high proliferation markers = HER2-driven phenotype
  without amplification -> favors HER2-directed therapy

**Practical constraint:** TCGA matched normals exist for only a subset; discordant
group is ~35. The intersection could be very small. Flag sample size.

**Additional quick win:** Check Fraction Genome Altered (FGA) for non-amplified
discordant. Genomically quiet = supports "normal-ish biology with incidental ERBB2";
genomically unstable = supports "disease-driven biology."

**Priority:** 2 of 9 (strongest mechanistic insight)

---

### Analysis #4: Clinical Correlates and Outcomes

**Question:** Are there statistically significant associations between discordant
groups and demographic/clinical/outcome variables?

**Part A -- Clinical correlates (non-survival):**
- Age, stage, grade, ER/PR status, histologic subtype, FGA
- Cleaned ER quantitative subscales
- `Cent17 Copy Number` and `HER2 cent17 ratio` -- directly relevant for
  polysomy 17, a known cause of IHC 3+/FISH- discordance. Check whether
  IHC+/RNA-low cases show elevated Cent17.
- `ER positivity scale used` -- could reveal whether ER quantification method
  varies systematically across sites/eras (confounder for #3).

**Part B -- Survival:**
- KM curves for discordant vs. concordant HER2-negative
- ~35 discordant patients split into CN-stratified subgroups = arms of ~15-20
- **Flag upfront as severely underpowered.** Wide CIs, log-rank will struggle
  to reach significance. Frame as hypothesis-generating, not definitive.

**Priority:** 5 (clinical correlates) and 8 (survival) of 9

---

### Analysis #5: Equivocal Resolution and CDx Validation (NEW)

This is the missed opportunity in the current analysis. The equivocal group stops
at "RNA can call HER2+/-" but has a cleaner, more immediately actionable CDx story.

#### 5a. Formal Concordance Table (HIGHEST PRIORITY)

RNA-predicted HER2 vs. FISH outcome in IHC 2+ patients.

Report: sensitivity, specificity, PPV, NPV. This is the table a regulatory strategy
team asks for first. If RNA achieves >90% concordance with FISH in the equivocal
population, that's a predicate-device-comparison argument for CDx filing.

NB02 Section 4.1 starts this (RNA AUC for FISH prediction in IHC 2+ subset) but
does not push to the full concordance table.

**Priority:** 1 of 9 (highest biopharma impact, most defensible, lowest effort)

#### 5b. Multi-Modal Concordance Tiers

Among equivocal patients, stratify by agreement across modalities (RNA, CN, FISH):
- Tier 1: All modalities agree -> high confidence reclassification
- Tier 2: Mixed signals -> candidates for additional testing or clinical follow-up

Characterize each tier biologically (expression profiles, pathway scores).

**Priority:** 7 of 9 (adds depth)

#### 5c. RNA Continuous Scoring for T-DXd Eligibility

After FISH resolution, many equivocal patients become HER2-low (IHC 2+/FISH-)
and are eligible for T-DXd (DESTINY-Breast04). T-DXd efficacy may correlate with
HER2 expression level within HER2-low.

RNA provides a continuous quantitative score where IHC provides only ordinal
categories. Show that RNA score stratifies the equivocal population into
biologically distinct subgroups -- setting up the hypothesis for future RWD analysis.

Cannot answer treatment-benefit question with TCGA (no treatment data), but can
show biological heterogeneity within the equivocal zone.

**Priority:** 4 of 9 (high strategic value)

#### 5d. Equivocal Demographics

Compare clinical/demographic features of equivocal-resolved-positive vs.
equivocal-resolved-negative. Are there enrichments that would help target a
prospective study?

**Priority:** 9 of 9 (if time permits)

---

## Priority Ranking Summary

1. **#5a** -- Equivocal concordance table (RNA vs. FISH in IHC 2+)
2. **#3** -- Normal tissue + ER pathway correlation for discordant biology
3. **#2** -- Prevalence estimation (reframed as molecular overexpression rate)
4. **#5c** -- Equivocal T-DXd spectrum / continuous scoring
5. **#4a** -- Clinical correlates (non-survival)
6. **#1** -- Testing method confounder (due diligence)
7. **#5b** -- Multi-modal concordance tiers for equivocal
8. **#4b** -- Survival analysis (hypothesis-generating, underpowered)
9. **#5d** -- Equivocal demographics

---

## Cross-Cutting Methodological Notes

### Label Training Sensitivity
The ML model used to score equivocal patients was trained on IHC labels, which
themselves contain noise (the whole point of the project). Consider a brief
sensitivity analysis: remove lowest-confidence training labels (contradiction-flagged
cases from NB01 label construction) and check if equivocal scoring changes materially.

### Polysomy 17 Check
`Cent17 Copy Number` and `HER2 cent17 ratio` are available in clinical data. For
IHC+/RNA-low discordant cases, elevated Cent17 would indicate polysomy rather than
true ERBB2 amplification -- a known mechanism for IHC/FISH discordance.

### Fraction Genome Altered (FGA)
Available in clinical data. Quick one-line analysis for non-amplified discordant
group: genomically quiet vs. unstable distinguishes "incidental ERBB2 expression"
from "disease-driven biology."

### ER Quantitative Scoring
Cleaned columns (`er_allred_score`, `er_hscore`, `er_intensity`,
`er_percent_positive`, `er_fmol_mg`) provide granular ER quantification beyond
binary ER+/-. Relevant for Analysis #3 (degree of ER-driven biology in
non-amplified discordant).

### Sample Size Discipline
- Discordant group: ~35 total, ~15-20 per CN stratum
- Equivocal group: ~18 in multimodal cohort (check actual N)
- IHC 2+ with FISH results: check actual N before committing to #5a
- Flag any subgroup analysis below n=15 as underpowered
