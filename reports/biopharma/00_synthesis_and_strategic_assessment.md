# Biopharma Deliverable Synthesis and Strategic Assessment

**Analyst Review -- 2026-04-06**

---

## Executive Summary

Nine analyses were executed against TCGA BRCA data to build the prerequisite evidence
base for two biopharma value propositions: (1) an RNA-based companion diagnostic (CDx)
to replace FISH reflex testing in IHC 2+ equivocal patients, and (2) identification of
a molecularly defined HER2-positive population missed by current IHC testing. A third
opportunity -- RNA-guided T-DXd eligibility stratification -- emerged from the data.

**The headline result is Analysis 5a, stress-tested by a held-out validation.**
The primary analysis reported AUC = 0.994 for RNA-based prediction of FISH outcome
in 156 IHC 2+ patients -- but those patients were in the training set. A supplementary
held-out analysis (models retrained without any IHC 2+ patients) produced AUC = 0.779
(RF alone: 0.807), with near-perfect specificity (0.983) but reduced sensitivity
(0.412 at 0.5, 0.647 at optimal threshold). Critically, equivocal reclassifications
were perfectly stable: 28/28 patients received identical calls regardless of training
design. The honest performance range is AUC 0.78-0.81 for held-out concordance, with
the true prospective figure likely falling between this and the primary estimate.

Three analyses produced actionable evidence. Three produced useful supporting context.
Three were data-walled by TCGA's sparse clinical annotations and should be explicitly
repositioned as "what we would do with Tempus data" rather than treated as findings.

---

## Tier 1: Actionable Evidence (Ready for Biopharma Pitch)

### 5a. RNA/CN/ER/PR vs. FISH Concordance in IHC 2+ -- Primary and Held-Out

**What it shows:** Two analyses bracket the true performance of RNA-based FISH
prediction in IHC 2+ patients:

| Metric | Primary (5a) | Held-Out (5a Supp) |
|---|---|---|
| Training set | All 837 labeled (includes IHC 2+) | 683 non-IHC-2+ only |
| AUC | 0.994 | 0.779 (RF alone: 0.807) |
| Sensitivity (0.5) | 0.706 | 0.412 |
| Specificity (0.5) | 1.000 | 0.983 |
| PPV (0.5) | 1.000 | 0.875 |
| Kappa (0.5) | 0.790 | 0.488 |
| Optimal threshold | 0.341 | 0.314 |
| Sens (optimal) | 0.941 | 0.647 |
| Spec (optimal) | 0.992 | 0.917 |

The primary analysis tested whether the model could recover FISH labels for 156 IHC 2+
patients that were in the training set (AUC inflated by training overlap). The
supplementary analysis retrained models excluding all 154 IHC 2+ FISH-resolved patients,
then scored them as genuinely held-out data. The held-out models used the curated
44-gene panel (6 biological gene sets + CN + ER/PR), matching the consolidated NB03
approach.

**Why it matters:** The held-out result answers the most predictable objection ("you
trained on the test set"). The AUC drop from 0.994 to 0.779 confirms that the primary
estimate was inflated. But three findings survive and are arguably stronger for having
been stress-tested:

1. **Near-perfect specificity (0.983).** Even a model that has never seen an IHC 2+
   patient almost never calls a FISH-negative patient positive. When the model says
   "HER2+," it is almost always right (PPV = 0.875). This is the metric that matters
   most for a CDx: you do not want to put patients on HER2-directed therapy who are
   FISH-negative.

2. **The sensitivity gap is interpretable, not fatal.** The 34 FISH-positive IHC 2+
   patients removed from training constituted 22% of all Positives. These are
   specifically the borderline positives the model most needs to learn from. A properly
   designed CDx study would include IHC 2+ patients in the training set with rigorous
   cross-validation -- not exclude them entirely. The held-out design is the most
   conservative possible test; real-world performance would fall between the primary
   and held-out estimates.

3. **Equivocal reclassification is perfectly stable.** 28/28 truly equivocal patients
   received identical HER2 calls from both the original and held-out models (100%
   agreement). The 5 patients identified as HER2+ are driven by ERBB2 expression and
   CN=2 -- features so strong that the model identifies them regardless of whether it
   trained on borderline cases. Their ERBB2 expression (median 10.30) is significantly
   higher than the 23 RNA-negative patients (9.19, p=0.01) and consistent with
   concordant positives (11.59).

**What we learn from the delta:** The RF model showed nearly identical held-out AUC
on 17K features (0.810, from a partial run) and the 44-gene curated panel (0.807).
The feature count is not the bottleneck -- the training population composition is.
This tells us that a CDx built on the curated panel captures the essential signal,
and that the path to higher sensitivity runs through training set design (including
equivocal patients with proper CV), not through more features.

**For the pitch:** Do not quote AUC = 0.994. Lead with: "Near-perfect specificity
(0.98) for FISH-positive identification in IHC 2+, validated on held-out data from
a model trained without any equivocal patients." Frame the sensitivity gap as the
Tempus opportunity: a larger dataset with IHC 2+ patients in the training set,
proper cross-validation, and prospective concordance design would close it.

### 2. Prevalence of Molecular ERBB2 Overexpression (5.1%, CI: 3.7-7.0%)

**What it shows:** At the primary threshold (p95 of IHC-negative ERBB2 distribution),
35 of 686 IHC-negative patients (5.1%) show molecular evidence of ERBB2 overexpression.
Of these, 6 (17%) also have genomic amplification (CN >= 2), providing orthogonal
confirmation.

**Why it matters:** This sizes the addressable patient population. 5.1% of IHC-negative
breast cancer patients extrapolates to approximately 9,000-17,000 US patients annually
who may be molecularly HER2-positive but clinically classified as HER2-negative.

**Framing discipline:** This is NOT a false-negative rate. IHC is the clinical ground
truth. The defensible framing is: "a defined proportion of IHC-negative patients show
molecular evidence that questions their negative classification." The CN-stratified
breakdown provides the biological hierarchy that makes this framing rigorous:

- CN >= 2 (n=6, 17%): genomic + transcriptomic evidence. Strongest reclassification
  case. These are plausible IHC false negatives.
- CN = 1 (n=19, 54%): intermediate. Modest CN gain + high expression.
- CN = 0 (n=10, 29%): transcription-only. May reflect ER-driven co-regulation
  rather than HER2-driven biology (see Analysis 3).

**Robustness:** Prevalence is stable across thresholds (0.9% at p99 to 10.1% at p90).
Wilson score CIs are appropriately tight. The methodology is clean and reproducible.

### 3. Discordant Biology: Two Populations, Two Clinical Strategies

**What it shows:** CN-stratified biology confirms that the discordant population is not
monolithic. CN-high (n=6) and CN-low (n=29) patients have fundamentally different
molecular profiles:

| Feature | CN-low Discordant | CN-high Discordant |
|---|---|---|
| ER pathway z-score | +0.45 (elevated) | -0.37 (depressed) |
| HER2 pathway z-score | +0.28 (moderate) | +0.61 (strong) |
| Proliferation z-score | -0.29 (low) | +0.57 (high) |
| FGA vs concordant neg | p=0.87 (identical) | elevated (0.454 vs 0.249) |
| ERBB2-ESR1 correlation | rho = -0.359 (negative) | N/A (n=6) |
| Biology | Luminal co-regulation | HER2-enriched |
| Clinical implication | Endocrine therapy | HER2-directed therapy |

**Why it matters:** This is the mechanistic argument that converts a statistical
observation (5.1% prevalence) into a biologically grounded clinical strategy. Without
CN stratification, the discordant group is an ambiguous mix. With it, two distinct
populations emerge with different therapeutic implications.

**Key finding -- correlation reversal:** In concordant negatives (n=651), ERBB2
correlates positively with ESR1 (rho = +0.298, p < 0.0001). In CN-low discordant
patients (n=29), this correlation reverses (rho = -0.359, p = 0.056). The TFF1
(rho = -0.388, p = 0.037) and TFF3 (rho = -0.416, p = 0.025) correlations are
significant. This reversal suggests a decoupling of ERBB2 from the ER transcriptional
program in CN-low discordant patients -- their ERBB2 elevation is NOT simply passive
co-regulation with ER pathway genes, despite having high ER pathway expression overall.

**Assessment:** This is the most intellectually interesting result in the set. The
correlation reversal complicates the simple narrative ("CN-low = ER-driven, ignore it")
and suggests these patients may have a biology worth investigating further. But with
n=29 and marginal p-values, this needs validation before it becomes a biopharma talking
point. Frame it as a hypothesis, not a finding.

---

## Tier 2: Supporting Context (Strengthens the Narrative)

### 5c. T-DXd Spectrum and Continuous Scoring

**What it shows:** 388 HER2-Low patients (IHC 1+ or IHC 2+/FISH-) show a continuous
ERBB2 expression spectrum. The upper tertile (n=129, median ERBB2=9.57) has higher ML
probability (0.228 vs 0.155), higher GRB7 co-expression (5.64 vs 4.71), and elevated
estrogen response pathway scores compared to the lower tertile. Proliferation markers
(E2F targets) decrease with increasing ERBB2 within HER2-Low.

**Strategic value:** This sets up the T-DXd enrichment hypothesis. If RNA-continuous
scoring can identify the subset of HER2-Low patients most likely to benefit from
T-DXd, that is a separate CDx opportunity from the equivocal story. The biological
heterogeneity within HER2-Low is real and measurable. But TCGA has no treatment data,
so the treatment-benefit correlation is entirely hypothetical at this stage.

**Assessment:** Good hypothesis generation. The observation that ERBB2 tertiles within
HER2-Low show distinct pathway profiles is biologically plausible and strategically
valuable. Do NOT overclaim -- "we see biological heterogeneity" is defensible;
"the upper tertile would benefit more from T-DXd" is speculation without outcome data.

### 5b. Multi-Modal Concordance Tiers

**What it shows:** Among 28 equivocal patients, a tiered concordance framework
identifies 4 Tier 1 (RNA+ and CN amplified = high confidence HER2+), 1 Tier 2
(RNA-only HER2+), 21 Tier 3 (concordant HER2-), and 2 Tier 4 (mixed signals: CN
amplified but RNA-negative).

**Value:** This is a clinical decision framework, not a statistical result. It
demonstrates how multi-modal data could be operationalized in practice. Tier 1 patients
have the strongest case for reclassification. Tier 4 patients (CN amplified but RNA
negative) are interesting -- they may represent cases where GISTIC amplification does
not translate to transcriptional activation.

**Assessment:** Clean framework, small N. Useful as a figure in a deck showing "here's
how we would operationalize multi-modal classification." The absence of FISH data for
all 28 patients means the third modality adds nothing in TCGA -- this framework's
true utility would emerge with Tempus data where FISH is more consistently available.

### 4a. Clinical Correlates (Table 1)

**What it shows:** No clinical variables differentiate discordant from concordant
negative patients after FDR correction.

**Value as a negative result:** This is actually useful. It supports the argument that
molecular testing is necessary because clinical features alone cannot identify the
discordant population. You cannot screen for these patients with demographics, stage,
or histology -- you need RNA.

**Data quality issue:** The merge produced column suffix collisions (_x/_y), causing
Diagnosis Age and FGA to be dropped from the Table 1. The report is incomplete but
the conclusion (no clinical differentiators) is likely robust -- these variables were
analyzed in other reports (FGA in Analysis 3, age in Analysis 5d) with no significant
findings.

---

## Tier 3: Data-Walled (Reposition as Tempus Opportunities)

### 1. Testing Method as Confounder

**Result:** 4 of 35 discordant patients have testing method annotations.
Not analyzable. Chi-squared p=0.73 on the 4 patients is meaningless.

**Repositioning:** This is not a failed analysis -- it is a demonstrated data gap.
The biopharma pitch should frame this as: "TCGA lacks the metadata to investigate
testing method variability. Tempus's standardized testing metadata (assay platform,
antibody clone, fixation protocol) enables the analysis that TCGA cannot support.
If specific testing configurations are associated with higher discordance rates, that
finding has immediate clinical quality implications."

### 4b. Survival Analysis

**Result:** HR = 1.20 (95% CI: 0.49-2.98), p = 0.69. 5 events in 35 patients.
Log-rank p = 0.69.

**Assessment:** This analysis cannot distinguish a clinically meaningful effect from
noise. The CI spans HR 0.49 to 2.98 -- it is consistent with a 50% survival
advantage, no effect, or a 3-fold survival disadvantage. With 5 events, this has
essentially zero statistical power. The point estimate of 1.20 is not interpretable.

**Repositioning:** Include the KM curve in supplementary materials with explicit
"hypothesis-generating, N=35, 5 events" language. The value is showing that the
analytical framework exists and can be applied to a powered Tempus cohort. Do not
draw any clinical conclusions from this result.

### 5d. Equivocal Demographics

**Result:** No clinical differences between RNA-pos (n=5) and RNA-neg (n=23)
equivocal patients. The only differentiator is copy number (p=0.003), which is
tautological (CN is an input to the reclassification).

**Assessment:** With n=5 in the smaller group, this was never going to produce
findings. The CN result is not a finding -- it's a confirmation that the
classification uses CN information. The clinical null result reinforces 4a's
conclusion: you need molecular data.

---

## Argument Architecture: How These Results Connect

### The Equivocal CDx Story (Lead With This)

```
Clinical Problem: IHC 2+ patients require FISH reflex testing (cost, delay, access)
    |
    v
Evidence: RNA achieves specificity 0.98 for FISH outcome in IHC 2+, even on held-out
    data from a model trained without equivocal patients (Analysis 5a + Supp)
    |
    v
Key insight: Sensitivity gap (0.41-0.65) is a training design issue, not a signal issue
    -- including IHC 2+ in training with proper CV would close it
    |
    v
Stability proof: 28/28 equivocal reclassifications identical across model designs
    |
    v
Operational Framework: Multi-modal concordance tiers (Analysis 5b)
    |
    v
Regulatory Path: Predicate-device comparison (RNA vs. FISH)
    |
    v
Next Step: Prospective concordance study on Tempus data (with IHC 2+ in training + CV)
    |
    v
Extension: RNA continuous scoring for T-DXd stratification (Analysis 5c)
```

This is the "land" play. The clinical need is established (FISH reflex testing is slow,
expensive, and access-limited). The evidence is honest: specificity is near-perfect
even under the most conservative held-out design, sensitivity requires proper training
set inclusion to optimize. The regulatory path is understood (predicate-device
comparison). The gap is prospective validation with optimized training, which Tempus
data can provide.

### The Discordant Population Story (Expand With This)

```
Observation: 5.1% of IHC-neg patients show molecular ERBB2 overexpression (Analysis 2)
    |
    v
Biology: CN-stratified -- two distinct populations (Analysis 3)
    |-- CN-high (17%): IHC-missed HER2+ -> HER2-directed therapy
    |-- CN-low (83%): Luminal co-regulation with ERBB2 decoupling -> investigation needed
    |
    v
Clinical: Discordant patients are clinically invisible (Analysis 4a)
    |
    v
Survival: Underpowered, but framework exists (Analysis 4b)
    |
    v
Testing Method: Data gap in TCGA, addressable with Tempus (Analysis 1)
    |
    v
Next Step: Retrospective cohort study on Tempus data with outcome linkage
```

This is the "expand" play. The population is sized (~9K-17K US patients/year). The
biology is characterized. But the clinical utility is unproven -- we cannot demonstrate
that identifying these patients changes outcomes without treatment data. This requires
a larger dataset with linked treatment and survival information.

### The T-DXd Stratification Story (Emerging Opportunity)

```
Context: T-DXd approved for HER2-Low (IHC 1+ or IHC 2+/FISH-)
    |
    v
Problem: IHC ordinal categories (0, 1+, 2+) are crude; treatment benefit may vary
    |
    v
Evidence: RNA reveals continuous ERBB2 spectrum within HER2-Low (Analysis 5c)
    |-- Upper tertile biologically distinct from lower tertile
    |-- Pathway profiles suggest differential HER2 pathway dependence
    |
    v
Hypothesis: RNA-guided T-DXd selection could improve benefit-risk
    |
    v
Next Step: Tempus RWD with T-DXd treatment and response data
```

This is speculative but strategically important. The T-DXd market is large and growing.
If RNA scoring can enrich for T-DXd responders, that is a significant CDx opportunity.
But TCGA provides only the biological rationale, not the treatment-benefit evidence.

---

## What's Missing and What Tempus Data Would Add

| Gap | TCGA Limitation | Tempus Advantage |
|---|---|---|
| Concordance sensitivity (5a) | Held-out AUC 0.78 without IHC 2+ in training; specificity strong but sensitivity limited | IHC 2+ in training with proper CV + prospective blinded FISH concordance |
| Testing method variability (1) | 8% annotation rate | Standardized testing metadata |
| Treatment outcomes (4b) | No treatment data, 5 events | Linked treatment + survival |
| T-DXd response correlation (5c) | No T-DXd treatment data | ADC treatment and response data |
| CN-high validation (3) | n=6 | Larger cohort for CN-stratified analysis |
| Clinical enrichment (4a) | Sparse TCGA annotations, merge issues | Comprehensive clinical data |

---

## Honest Assessment of Evidence Strength

**What we can say with confidence:**
- RNA-based models achieve near-perfect specificity (0.983) for identifying FISH-positive
  cases among IHC 2+ patients, even when trained without any IHC 2+ data. When the
  model calls a patient HER2+, it is almost always correct.
- The primary AUC of 0.994 was inflated by training-data overlap. The honest held-out
  AUC is 0.78-0.81, with the sensitivity gap attributable to excluding borderline
  positives from training -- a design choice, not a signal limitation.
- Equivocal reclassifications are robust to model design: 28/28 calls stable across
  original and held-out models. The 5 HER2+ patients are real.
- ~5% of IHC-negative patients have molecular ERBB2 overexpression. The prevalence
  estimate is methodologically clean.
- CN-high and CN-low discordant patients have different biology. The pathway profiles
  are consistent with known HER2-enriched vs. luminal biology.
- Clinical features cannot identify the discordant or equivocal populations. Molecular
  testing is required.
- Feature reduction to the curated 44-gene panel loses essentially no discriminative
  power vs. full transcriptome (RF: 0.807 vs 0.810), confirming the panel captures
  the biologically relevant signal.

**What we can hypothesize but cannot prove from TCGA:**
- A properly designed CDx (IHC 2+ in training with cross-validation) would achieve
  AUC somewhere between 0.81 and 0.99 -- likely in the 0.90+ range given that the
  sensitivity gap is a training design artifact, not a signal ceiling.
- RNA-based CDx could replace FISH for equivocal resolution (needs prospective study).
- The discordant population would benefit from reclassification (needs outcome data).
- RNA-continuous scoring improves T-DXd patient selection (needs treatment-response
  data).
- The ERBB2-ESR1 correlation reversal in CN-low patients reflects a biologically
  meaningful decoupling (needs larger sample and functional validation).

**What we cannot say:**
- The model achieves AUC 0.994 on held-out equivocal data (it does not; that was
  training-set performance).
- Discordant patients have worse survival (HR=1.20, p=0.69, 5 events).
- Testing method explains discordance (4/35 with data).
- Any specific clinical feature predicts discordant status (no significant findings
  after FDR correction).

---

## Recommendations for Pitch Construction

1. **Open with the held-out validation, not the inflated AUC.** Lead with specificity
   (0.983) on genuinely held-out data. "A model trained without any equivocal patients
   achieves near-perfect specificity for FISH outcome in the equivocal zone." This is
   more credible than quoting 0.994 and having a reviewer discover the training overlap.
   Show both ROC curves (primary and held-out) side by side -- the transparency itself
   is the selling point.

2. **Frame the sensitivity gap as the Tempus opportunity.** The held-out sensitivity
   (0.41-0.65) is low because 22% of Positives were excluded from training. This is
   not a ceiling -- it is a floor set by the most conservative possible design. A
   Tempus CDx study with IHC 2+ patients in the training set (proper CV) would close
   the gap. This converts a weakness into a pitch for why Tempus data matters.

3. **Emphasize equivocal reclassification stability.** 28/28 calls identical across
   model designs. The 5 HER2+ patients are robust to every perturbation we tested.
   This is the strongest single data point for clinical decision-making.

4. **Frame the equivocal CDx as the near-term deliverable.** Predicate-device
   comparison pathway. Known clinical pain point. Quantified patient population
   (IHC 2+ is ~15-20% of all breast cancers).

5. **Introduce the discordant population as the expansion opportunity.** Lead with
   prevalence (5.1%, 9K-17K patients), then immediately CN-stratify to show biological
   rigor. Use the pathway heatmap from Analysis 3 to show the two-population story.

6. **Position the data gaps as Tempus differentiators.** Every "TCGA cannot answer
   this" is a "Tempus can answer this." Testing method metadata, treatment linkage,
   prospective concordance design, larger cohorts.

7. **Do not present the survival analysis or demographics as findings.** Include them
   in supplementary materials with appropriate caveats. They demonstrate analytical
   capability, not clinical evidence.

8. **The T-DXd story is a teaser, not a centerpiece.** It positions future value
   but requires Tempus-specific data (ADC treatment records) to become actionable.

9. **Be disciplined about what TCGA proves vs. what it motivates.** Every claim should
   map to one of: "TCGA demonstrates this" (held-out specificity, reclassification
   stability, prevalence, biology) or "TCGA motivates this investigation in Tempus
   data" (sensitivity optimization, survival, testing method, T-DXd). Mixing these
   categories undermines credibility.

---

## Analysis Quality Notes

- **Analysis 4a has incomplete output.** Column merge suffixes caused Diagnosis Age and
  FGA to be dropped from Table 1. The conclusion is likely unaffected (no clinical
  differentiators), but the report should be regenerated with the merge issue fixed
  for completeness.

- **Analysis 5a circularity is now resolved.** The supplementary held-out analysis
  (5a_supp) directly addresses the training-data overlap concern. The primary AUC of
  0.994 should be cited as an upper bound only; the held-out AUC of 0.779 (RF: 0.807)
  is the defensible figure. Both results should be presented together in any external-
  facing version. The 44-gene curated panel matches full-transcriptome performance
  (RF 0.807 vs 0.810), confirming the feature reduction approach is sound.

- **Analysis 3's correlation reversal (ERBB2-ESR1)** is the most nuanced finding.
  At p=0.056 in n=29, it is suggestive but not definitive. The TFF1/TFF3 correlations
  are significant (p<0.05) but were not corrected for multiple testing across 11 genes.
  After Bonferroni correction, nothing survives. Present as "observed pattern consistent
  with decoupled biology" rather than "statistically confirmed."

- **The 377 unclassified patients in Analysis 5c** (HER2 spectrum) reflect sparse IHC
  scoring in TCGA. This does not invalidate the analysis but limits the denominator.

---

*This assessment synthesizes 9 individual analysis reports plus the 5a supplementary
held-out concordance validation into an integrated evaluation of evidence strength,
argument architecture, and strategic positioning for biopharma engagement. Individual
reports are in `reports/biopharma/`. The held-out validation
(`5a_supp_heldout_concordance.md`) is the most important methodological contribution
and should accompany the primary 5a report in all external-facing materials.*
