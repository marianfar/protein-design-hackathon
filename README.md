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
| Celia | Evidence-guided seed sequences plus constrained AbLang2 framework substitutions | Boltz-2 and custom ipSAE screen | 30 candidates and WT scored |
| FULL_PIPELINE | Structure-contextual ProteinMPNN proposals, AbLang2 filtering, Boltz-2 folding | Combined ipSAE and AbLang2 ranking | Integrated implementation; first-run checks documented as pending |

These workflows should be compared as complementary strategies, not treated as interchangeable runs. In particular, the three DSM candidates are not automatically part of Celia's 30-candidate panel, and the integrated FULL_PIPELINE shortlist is not automatically the same as either set.

---

## 4. Reproducing the Workflows

Each workflow has its own working directory and dependencies. Run commands from the directory containing that workflow's scripts.

### 4.1 DSM

From the repository root, with Modal installed and authenticated:

```bash
modal run modal_app.py
```

Generation settings, including the number of candidates, are configured in `src/config.py`.

### 4.2 Celia: AbLang2 Design and Boltz-2 Evaluation

Follow the instructions in the Celia workflow README. The documented process generates the AbLang2 shortlist locally and runs Boltz-2 on Modal. For the reported 30-candidate evaluation, the documented command is:

```bash
modal run modal_eval.py --limit 30
```

The required reference and shortlist CSV files must be present in the expected `ablang2_results/` directory. See that workflow's README for the full setup, scoring, and score-only retrieval commands.

### 4.3 FULL_PIPELINE

From the FULL_PIPELINE directory, after installing and authenticating the Modal CLI:

```bash
pip install -U modal
modal setup
modal run pipeline.py --stage check
```

The smoke test should be run before launching the full workflow. If it passes, the documented full run is:

```bash
modal run pipeline.py
```

Settings can be changed in the `CONFIG` block or overridden for a single run:

```bash
modal run pipeline.py --overrides "n_samples=20;n_fold=6;chunk=3"
```

Check Modal spending controls before launching GPU jobs. GPU timeouts and sample counts can affect total costs.

---

## 5. Interpretation and Limitations

- **Predicted scores are not experimental binding affinity.** AbLang2 likelihood/PLL, Boltz-2 ipTM, and ipSAE are model-based sequence or structural confidence measures. They do not directly measure K_D, kinetics, or therapeutic efficacy.
- **The workflows use different design assumptions.** DSM generation, evidence-guided AbLang2 design, and ProteinMPNN structure-contextual redesign do not produce directly comparable candidate populations.
- **Experimental evidence has context.** Measured multi-mutant sequences and literature-reported mutation effects are assay- and construct-specific. Effects should not be assumed additive or directly comparable to the organizer's WT measurement.
- **Predicted interfaces need inspection.** A high interface-confidence score does not by itself establish the correct binding pose or specificity.
- **Developability remains to be established.** Expression, stability, solubility, aggregation, specificity, and other properties require further evaluation.
- **No new variant is experimentally validated in this project based on the documented computational results.** Laboratory testing is required to determine whether any candidate improves HER2 binding.

---

## 6. Overall Project Status

| Work item | Status |
|---|---|
| HER2 target and parent scFv preparation | Completed |
| DSM generation strategies and three candidate sequences | Completed |
| DSM candidate sequence-integrity checks | Passed |
| Evidence-guided AbLang2 design and 30-sequence shortlist | Completed |
| Boltz-2/ipSAE evaluation of Celia's 30 candidates and WT | Reported complete |
| Final selection of 15 unique candidates | Pending |
| FULL_PIPELINE implementation | Documented; first-run verification required |
| Experimental validation of new variants | Not yet completed |

---

## 7. Key References and Tools

- [DSM implementation — Gleghorn Lab](https://github.com/Gleghorn-Lab/DSM)
- [AbLang2 implementation](https://github.com/oxpig/AbLang2)
- [Boltz implementation](https://github.com/jwohlwend/boltz)
- [Dunbrack Lab ipSAE](https://github.com/DunbrackLab/IPSAE)
- [HER2–trastuzumab Fab structure, PDB 1N8Z](https://www.rcsb.org/structure/1N8Z)
- [Human HER2/ERBB2, UniProt P04626](https://www.uniprot.org/uniprotkb/P04626)
- [A-Alpha Bio open AlphaSeq dataset](https://huggingface.co/datasets/aalphabio/open-alphaseq)

---

*This repository documents computational design and prioritization work for a hackathon. It does not establish that any newly generated antibody variant has improved experimental HER2 affinity or therapeutic performance.*
