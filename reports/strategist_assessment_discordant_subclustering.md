# Strategist Assessment: Discordant Subclustering & Narrative Tightening

**Date:** 2026-04-07

---

## Problem Statement

The Tempus coding challenge asks to "find patient subsets that appear biologically
or clinically distinct and align with HER2 status" via unsupervised learning. The
current analysis performs cohort-level k=4 clustering (NB02 Section 6) but defines
the equivocal and discordant subgroups by clinical rules and thresholds rather than
discovering them through unsupervised structure. This creates a gap between the
prompt's emphasis on subset discovery and the analysis's clinical-rule-based subgroup
definitions.

A secondary tension: the prompt asks whether RNA or DNA is more predictive of
clinical IHC. NB02 answers this with single-gene ERBB2 ROC, and NB03 answers it
with panel-based ML, but the narrative bridge between these two levels of evidence
is not explicit.

## Recommendation

### 1. Discordant Group Subclustering (Do)

Perform unsupervised clustering on the 35 IHC-negative/RNA-high discordant patients
using the full filtered transcriptome. Purpose: independently validate or refine the
CN-stratified two-population finding (NB03 Section 5) through an orthogonal analytical
approach.

**Why full transcriptome, not curated panel:**
- Unsupervised clustering on the curated panel is circular -- the panel was designed
  to capture HER2 biology, so it will trivially separate CN-high from CN-low.
- Full-transcriptome clustering is genuinely exploratory. If it independently recovers
  the CN split, that's strong evidence the two populations are transcriptome-wide
  distinct, not just different on one axis.
- The circularity concern that applies to supervised ML (label leakage through feature
  selection) does not apply to unsupervised methods. There are no labels in PCA + k-means.

**Expected outcome:** k=2 clusters that align with CN status, confirming the
CN-stratified biology is not an artifact of a single variable but reflects genuine
transcriptomic divergence.

**Placement:** NB03 Section 5, between the current CN stratification and the consensus
molecular HER2 score. Narrative: CN stratification (observation) -> unsupervised
confirmation (validation) -> consensus score (integration).

### 2. RNA-vs-DNA Narrative Bridge (Do)

Add 2-3 sentences in NB03 Section 2 (or its intro) explicitly connecting the
single-gene ERBB2 ROC from NB02 to the panel-based ML comparison. Frame it as:
"NB02 established RNA > CN at the single-gene level; here we ask whether this
advantage extends to multi-gene panels and more complex models."

**Effort:** Framing only, no new analysis.

### 3. Equivocal Group Subclustering (Skip)

Already well-addressed by concordance analysis, reclassification stability, and
multi-modal tiers in NB04 Part I. At n=28 truly equivocal patients, subclustering
would be noisy and add little beyond what the existing framework provides.

## Priority

1. Discordant subclustering (20-30 min, strengthens central finding)
2. RNA-vs-DNA narrative bridge (5 min, framing fix)
3. Equivocal subclustering (skip)
