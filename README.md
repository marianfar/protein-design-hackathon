# Computational Design and Optimization of Trastuzumab scFv Against HER2

**Adaptyv Bio Zurich Hackathon · Computational Antibody Engineering · 31 October 2026**

This repository brings together three complementary computational approaches developed by the team to explore and prioritize trastuzumab-derived single-chain variable fragment (scFv) variants targeting human HER2/ERBB2:

1. **Initial sequence-generation exploration using DSM** — an exploratory effort to generate variants by masking selected antibody regions.
2. **Evidence-guided design and structural screening using AbLang2 and Boltz-2** — a constrained design workflow incorporating experimentally characterized sequences and literature-derived mutation evidence.
3. **An integrated ProteinMPNN–AbLang2–Boltz-2 pipeline** — a more automated workflow designed to propose, score, fold, and shortlist candidate variants.

These are related but distinct workflows. They do not represent one uninterrupted sequence of model stages: each has its own design strategy, inputs, scripts, outputs, and validation status. This README distinguishes completed computational results from workflows that are implemented but still require execution or verification.

---

## 1. Project Objective

The challenge is to improve a trastuzumab-derived scFv that binds human HER2. The organizer-reported wild-type (WT) binding affinity is **0.82 nM K_D**. The design task includes preserving the scFv architecture, meeting sequence constraints, and selecting a final set of 15 unique candidates for experimental testing.

The common construct used across the project is:

| Component | Length | Position in full scFv |
|---|---:|---:|
| Variable heavy chain (VH) | 120 aa | 1–120 |
| Flexible linker, `(GGGGS)3` | 15 aa | 121–135 |
| Variable light chain (VL) | 107 aa | 136–242 |
| **Total** | **242 aa** | **1–242** |

The HER2 target for structural evaluation is its extracellular domain: UniProt **P04626**, residues **23–652**, a 630-aa sequence.

The principal sequence and structure reference is the trastuzumab-derived antibody structure **PDB 1N8Z**.

---

## 2. Project History and Three Complementary Workflows

### 2.1 Initial Exploration: DSM Sequence Generation

**Purpose:** Test whether a pretrained protein–protein interaction model could generate alternative scFv sequences when selected antibody regions were masked.

The DSM workflow used the pretrained `Synthyra/DSM_ppi_full` model. Three mutation strategies were implemented:

| Strategy | Region | Masked positions | Generated candidate |
|---|---|---:|---|
| All six CDRs | VH and VL CDRs | 65 | `all_six_CDRs_candidate_1` |
| Light-chain loop | VL local positions 66–71 | 6 | `light_chain_loop_candidate_1` |
| Heavy-chain CDR2 loop | VH local positions 52–57 | 6 | `heavy_chain_CDR_loop_candidate_1` |

The generated sequences were reconstructed as full-length VH–linker–VL scFvs. The initial run generated one candidate per strategy.

**Validation performed:** All three candidates were 242 amino acids long, contained canonical amino acids only, and retained the original 15-aa linker. They contained 58, 6, and 5 substitutions, respectively, relative to the parent sequence.

**Implementation:**
- `src/config.py`
- `src/sequence_utils.py`
- `src/mutation_strategies.py`
- `src/dsm_generator.py`
- `modal_app.py`
- DSM implementation in `external/DSM/`

Generation was executed using Modal GPU infrastructure. A Transformers compatibility pin and a FastPLMs module-path compatibility workaround were used in the Modal environment.

**Status:** DSM generation completed and sequence-integrity checks passed. These candidates were exploratory outputs; their HER2 binding, structural quality, and developability were not established by sequence generation. They should not be confused with the separate 30-candidate AbLang2/Boltz-2 panel described below.

### 2.2 Evidence-Guided Design and Screening: Celia Workflow

**Purpose:** Use experimental and literature-derived sequence evidence to build a constrained candidate panel, then compare predicted HER2–scFv complex confidence.

This workflow used the paired **AbLang2** antibody language model and **Boltz-2** complex structure prediction.

The design inputs included:

- The challenge WT scFv.
- Five experimentally characterized AlphaSeq-derived multi-mutant sequences (`V1`–`V5`).
- Literature-associated affinity and stability mutation evidence.
- Four rationally combined, untested designs (`B1`–`B4`).
- AbLang2-generated framework variants derived from selected seed sequences.

The AbLang2 design script scanned five seed backgrounds (`V1`, `V2`, `V3`, `V5`, and `SPR4`) at 24 allowed framework positions. It used masked log-odds to propose substitutions, followed by sequence-quality filters and paired pseudo-log-likelihood (PLL) scoring. A diversity-aware selection process produced 21 novel ML combinations. These were combined with five measured AlphaSeq sequences and four rational designs to form a 30-sequence structural screening panel.

Boltz-2 was used to predict complexes between each scFv and the 630-aa HER2 extracellular domain. The reported screen used empty MSAs and one diffusion sample per complex. The custom evaluator extracted inter-chain PAE information and calculated directional ipSAE scores, including `ipsae_min`.

**Reported computational result:** All 30 candidate rows and the separate WT reference were successfully scored. Eight candidates had `ipsae_min` values above WT in this screen. The highest-scoring candidate was `ML_20` (`ipsae_min = 0.3340`; WT = `0.1145`). Seven of the eight WT-beating candidates also exceeded WT on ipTM.

These scores represent predicted structural/interface confidence, not experimental affinity. In particular, a score approximately 2.92 times the WT ipSAE value does **not** mean a 2.92-fold improvement in binding affinity.

**Main implementation and outputs** (inside the Celia workflow folder):

- `trastuzumab_ablang2_design.py` — sequence proposal, filtering, and shortlist generation.
- `eval.py` — Boltz-2 input preparation and ipSAE/confidence scoring.
- `modal_eval.py` — Modal execution and result retrieval.
- `ablang2_results/` — reference sequences, proposals, rankings, finalists, and the 30-candidate shortlist.
- `boltz_results/` — WT and candidate scoring outputs and, where retrieved, predicted structures and confidence files.
- `docs/` — literature/evidence and sequence-generation documentation.

**Status:** The 30-candidate Boltz-2/ipSAE screen is reported as complete. The final 15 candidates still require selection, and no newly designed variant has been experimentally validated in this project.

### 2.3 Integrated Workflow: FULL_PIPELINE

**Purpose:** Provide a configurable, automated pipeline that connects candidate proposal, antibody sequence scoring, complex prediction, and final ranking.

The newer integrated workflow is organized into two main files:

- `envs.py` — defines the separate Modal environments, images, volumes, and remote functions.
- `pipeline.py` — controls the stages, filters, ranking, and run configuration.
- `her2_pipeline.py` — an earlier single-file version retained as a backup.

Its three stages are:

| Stage | Tool | Role | Configured GPU |
|---|---|---|---|
| 1. Propose | ProteinMPNN / AbMPNN | Redesigns antibody residues near HER2 while keeping HER2 fixed as context | L4 |
| 2. Score | Paired AbLang2 | Compares proposed residues with their parent residues using masked scoring | L4 |
| 3. Fold and rank | Boltz-2 + Dunbrack ipSAE | Predicts scFv–HER2 complexes and calculates interface confidence | A100 80 GB |

By default, designable antibody positions are selected using proximity to HER2, with configurable contact cutoffs and optional explicit VH/VL design and fixed-position lists. Cysteines can be protected. Candidates are filtered by mutation count and optional AbLang2 thresholds before a subset is sent to Boltz-2. The parent is folded as a baseline. A weighted ranking combines ipSAE and AbLang2 delta to support selection of a final 15.

Run settings are collected in the `CONFIG` block in `pipeline.py` and saved as `config.json` in each timestamped run directory.

**Expected outputs:**

- `1_candidates_all.csv` — all ProteinMPNN proposals.
- `1_candidates.csv` — candidates retained after mutation filtering.
- `2_scored.csv` — AbLang2-scored candidates.
- `3_folded.csv` — Boltz-2 results, including the parent baseline.
- `4_shortlist.csv` — final weighted shortlist.
- `config.json` — settings used for that run.
- Predicted structures and confidence files — stored in the Modal volume `her2-results`.

**Status:** The integrated workflow is implemented and documented, but its README explicitly lists first-run checks that still need verification. These include checkpoint compatibility, FASTA chain matching, AbLang2 mask-token behavior, Boltz-2 output/PAE compatibility, and pinning the exact Boltz version after a successful run. Therefore, it should be treated as the most integrated workflow design, not as a completed or experimentally validated result unless its run outputs confirm that status.

---

## 3. How the Approaches Relate

The three efforts explore different ways to reach the same goal:

| Workflow | Main design idea | Main evaluation | Evidence/status |
|---|---|---|---|
| DSM | Mask selected CDRs or loops and generate sequence alternatives | Sequence-integrity checks completed; downstream structural evaluation is separate | Three candidates generated |
| Celia | Evidence-guided seed sequences plus constrained
