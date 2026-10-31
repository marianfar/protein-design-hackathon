# Trastuzumab scFv sequence-generation pipeline

**Project:** Adaptyv Bio Zurich Hackathon — trastuzumab/HER2 optimization  
**Implementation:** `trastuzumab_ablang2_design.py`; downstream `eval.py` and `modal_eval.py`  
**Documented:** 31 October 2026  
**Status:** Sequence generation and one Boltz-2 screen of 30 candidates completed; no new designs experimentally validated yet.

## 1. Executive summary

We constructed a **constrained, evidence-seeded antibody-design pipeline** rather than mutating the entire trastuzumab sequence at random.

1. Begin with the organizer's **exact 242-residue trastuzumab scFv** and five experimentally measured AlphaSeq variants, together with published HCDR3/stability findings.
2. Retain five measured sequences (`V1`–`V5`), build four five-mutation **rational combinations** (`B1`–`B4`), and use a four-mutation composite (`SPR4`) as an additional language-model seed.
3. Choose **five scan seeds** (`V1`, `V2`, `V3`, `V5`, `SPR4`) and **24 hand-curated framework positions**. At each position, let **paired AbLang2** score single amino-acid substitutions in their full VH/VL sequence context. **Do not freely mutate CDRs, the main HER2-contact residues, disulfide cysteines or the linker.**
4. Generate **up to 360 one-substitution proposals** (5 seeds × 24 positions × top 3 candidate residues), reject designs violating structural/developability constraints, then narrow to **up to 40** preselected candidates, **up to 30** full pseudo-log-likelihood finalists, and **21** diverse AbLang2 variants.
5. Assemble **30 scFv candidates**: five previously measured AlphaSeq sequences + four rational combinations + 21 new, untested AbLang2 extensions. All remain **242 amino acids** and have **5–7 substitutions relative to the challenge parent**.
6. Cofold each candidate with the **630-residue HER2 ectodomain (Thr23–Thr652)** using Boltz-2 on a Modal GPU. Compute **ipSAEmin** from inter-chain predicted-aligned-error matrices and compare with a separately predicted WT baseline.

**Scientific meaning:** The language model was used for **antibody sequence plausibility**, not affinity prediction. The subsequent Boltz-2/ipSAE ranking is a **structural-confidence screen**, not an experimental binding measurement. The ultimate selection for lab testing should combine these scores with evidence, diversity, structural inspection and developability checks.

## 2. Design objective and hackathon constraints

The organizers supplied a trastuzumab scFv that binds HER2 and report a **parent K_D of 0.82 nM by BLI on their platform**. The submission must preserve the **VH–(GGGGS)3–VL chain order**, have **at least five amino-acid substitutions relative to that parent** and be **no longer than 260 residues**. Teams may submit **15 sequences** for wet-lab assessment (the 30-sequence computational shortlist here is **not** that submission).

We additionally imposed stricter **project-specific** constraints: exactly **242 residues**, unchanged linker, unchanged disulfide-related cysteines and selected antigen-contact positions, **at most seven** total mutations, and only **one newly generated framework substitution per experimental/composite seed**. These stricter limits are engineering choices; they are not additional rules imposed by the organizer.

### 2.1 Exact parent construct

```text
VH (120 aa)
EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS

Linker (15 aa)
GGGGSGGGGSGGGGS

VL (107 aa)
DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK
```

Numbering throughout is **1-based in the concatenated 242-aa scFv**, not IMGT/Kabat. A full-scFv position `p` in VL corresponds to **VL-local residue `p − 135`**. Thus `S211T` is **VL residue 76**. The VH occupies 1–120, the linker 121–135 and the VL 136–242. **No indels** were proposed, so length never varies.

## 3. Evidence gathering and seed construction

We combined **three evidence categories**, carefully tracking what was actually measured. Details and citations are recorded in [`LITERATURE_AND_EXPERIMENTAL_EVIDENCE.md`](LITERATURE_AND_EXPERIMENTAL_EVIDENCE.md).

### 3.1 Five measured AlphaSeq variants (`V1`–`V5`)

The following full sequences were represented as mutations relative to the organizer's supplied scFv parent. **The complete combinations** were measured in the AlphaSeq datasets; it is not established that each constituent substitution is beneficial in isolation.

| Seed | Count | Parent-relative mutations | Source | Role in our pipeline |
|---|---:|---|---|---|
| `V1` | 6 | `T54S, G66D, T74I, Q82E, D102A, M107F` | `YM_0990` | Fixed shortlist design **and** AbLang2 seed |
| `V2` | 5 | `Y52S, K65Q, S75D, K76S, M107F` | `YM_0990` | Fixed shortlist design **and** AbLang2 seed |
| `V3` | 5 | `N55D, V93T, D102Y, M107F, D108Q` | `YM_0990` | Fixed shortlist design **and** AbLang2 seed |
| `V4` | 5 | `N55D, E89D, V93T, D102Y, M107F` | `YM_0990` | **Fixed shortlist only**; not an AbLang2 scan seed |
| `V5` | 5 | `G100L, G101I, D102G, A106D, M107F` | `YM_0989` | Fixed shortlist design **and** AbLang2 seed |

Why: these designs already contain the hackathon-required number of mutations and provide experimentally informed starting points. Using multiple distinct seeds diversifies sequence backgrounds and reduces reliance on one hypothesized binding mechanism.

### 3.2 Published/curated references not directly eligible for submission

| Reference | Mutations | Use | Why not directly included in the 30 |
|---|---|---|---|
| `WT` | None | WT prediction and baseline ipSAE | 0 mutations |
| `SPR2` | `S97A, D102Y` | Literature/SPR-related HCDR3 control | Only 2 mutations |
| `STABILITY2` | `S85N, R87T` | Published framework stability reference | Only 2 mutations; creates potential `N-L-T` glycosylation sequon |
| `SPR4` | `S97A, D102Y, S75D, R87T` | **Computational seed** for one additional framework edit | Only 4 mutations; untested combination |

`SPR4` combines the two HCDR3 substitutions recorded from an Absci-associated SPR evidence stream, `S75D` observed within measured AlphaSeq `V2`, and `R87T` borrowed *without S85N* from a published stability double mutant. **No source demonstrates that SPR4 itself has improved affinity.** `SPR4` was used to generate *five-mutation* ML candidates; it is not a direct final candidate.

### 3.3 Four rational five-mutation designs (`B1`–`B4`)

The script created four **new, untested** extensions of `SPR4` to satisfy the ≥5-mutation rule:

| Candidate | Fifth change relative to SPR4 | Full mutations |
|---|---|---|
| `B1` | `Y52S` | `Y52S, S75D, R87T, S97A, D102Y` |
| `B2` | `G101A` | `S75D, R87T, S97A, G101A, D102Y` |
| `B3` | `G100L` | `S75D, R87T, S97A, G100L, D102Y` |
| `B4` | `T54D` | `T54D, S75D, R87T, S97A, D102Y` |

`B1` inherits `Y52S` from the measured `V2` sequence; `B3` uses `G100L` from `V5`; `B2`/`B4` hypothesize different substitutions at CDR sites. **A mutation appearing in a measured sequence does not imply that this new composite is experimentally validated.** These four were defined **manually** as rational designs, not generated by AbLang2.

## 4. Why the AbLang2 search was deliberately restricted

An antibody sequence model can suggest amino acids that resemble its training distribution, but it does not explicitly optimize a specific HER2-binding epitope or anticipate every structural failure. The pipeline therefore **fixed the existing evidence-guided CDR mutations and searched only relatively conservative framework sites**.

The whitelist is a **precomputed/manual selection hard-coded into the script**, motivated by an inspection of PDB 1N8Z and documented approximate criteria (relative solvent accessibility ≥25%, distance to HER2 ≥10 Å, distance to opposing variable domain ≥8 Å), plus conservative exclusions. **The script does not recalculate solvent accessibility or distances at runtime**, so those cutoffs are design rationale, not runtime-enforced measurements. Consider them approximate until the structural selection is independently reproduced.

### 4.1 The exact 24 framework positions available to AbLang2

| Domain | Allowed full-scFv positions | Number |
|---|---|---:|
| VH | `7, 13, 17, 19, 23, 63, 69, 71, 72, 84, 88` | 11 |
| VL | `142, 144, 145, 147, 150, 152, 153, 155, 195, 198, 211, 212, 216` | 13 |
| **Total** | | **24** |

In particular, the model **could not invent new CDR substitutions**. CDR substitutions in the final designs come from the measured/rational **starting seeds**, where deliberately introduced and documented. The 24 whitelist positions do not overlap with any of the hard-locked lists below.

### 4.2 Protected positions and other restrictions

**Hard-coded HER2 contact locks**, based on the design's PDB 1N8Z analysis:

```text
33, 50, 57, 58, 59, 99, 103, 104, 105,
163, 164, 165, 166, 167, 184, 185, 188, 201,
226, 227, 228, 229
```

**Disulfide-preserving cysteine positions:** `22, 96, 158, 223`. The entire linker (`121–135`) is also fixed.

**Evidence-only sites excluded from new language-model mutation discovery:**

```text
52, 54, 55, 65, 66, 74, 75, 76, 82, 85, 87, 89, 93,
97, 100, 101, 102, 106, 107, 108
```

**Approximate CDR masks** used by code (full-scFv positions): VH 26–33, 51–58, 97–109; VL 162–167, 185–187, 224–232. Their purpose is to *exclude de novo model edits* to those sites; these ranges are **not** a formal IMGT/Kabat annotation.

**Candidate quality-control (`validate_candidate`):**

- Must be **242 amino acids**, using the 20 standard amino acids; no insertion or deletion; unmodified `(GGGGS)3` linker.
- Must retain the protected contacts, native disulfide cysteines and **total cysteine count**.
- Must have **5–7 total mutations** relative to the original hackathon parent (default max 7).
- Must preserve **every mutation already present in its seed**: AbLang2 may **add one edit**, never revert or overwrite a seed edit.
- The newly edited position must belong to the **24-position framework whitelist**.
- Do **not** introduce `C`, `P`, `G` or `W` at a newly edited framework position. This is a **conservative heuristic to limit liabilities**, not a claim that those residues are universally unfavorable.
- Do **not** introduce new `N-X-S/T` glycosylation sequons (where `X ≠ P`), either relative to the seed or parent.

The rational/experimental reference table is also checked for parent-relative mutation counts and protected-site constraints (except explicitly marked reference-only controls).

## 5. Exact model-generation procedure

The script uses the authors' **paired AbLang2** checkpoint (`ablang2-paired`) [AbLang2 repository](https://github.com/oxpig/AbLang2). It is an antibody **masked language model**, not a HER2 affinity predictor.

### 5.1 Encode the antibody as a paired variable-domain input

For AbLang2, **remove only the 15-aa linker**, and supply the heavy and light variable-domain sequences as a paired input: `[VH, VL]`. The candidate sequence used for downstream structure prediction still includes the complete 242-aa **VH–linker–VL** construct.

At a selected position `p`, replace **one existing amino acid** with the AbLang2 mask token `*`. The script accounts for AbLang2's internal paired-chain delimiters and verifies the mask token index (different offsets in VH versus VL). This avoids mistakenly applying a proposed VL mutation at a VH position or in the linker.

### 5.2 Score and propose a single substitution at each permitted position

For each of the **five seeds × 24 positions**:

1. Mask the target residue, leaving the remaining VH and VL sequence untouched.
2. Run AbLang2 to obtain the 20-amino-acid likelihood distribution at that location.
3. Calculate the model's **masked log-odds** of substituting amino acid `a` rather than keeping the original seed residue `w`:

   ```text
   masked_log_odds(a at p) = log P_AbLang2(a | masked VH/VL context)
                           − log P_AbLang2(w | masked VH/VL context)
   ```

4. Exclude the original residue and any newly introduced `C/P/G/W`; sort the remaining amino acids by masked log-odds.
5. Take the **top 3 amino-acid proposals for that position** (CLI default `--top-aa-per-position 3`).
6. Generate one **full 242-aa sequence** for each proposed single edit, keeping all seed mutations fixed.
7. Apply all quality-control filters, remove duplicate full sequences, and write the survivors to `01_all_model_proposals.csv`.

The **maximum pre-filter proposal space is `5 × 24 × 3 = 360 sequences`**. This is the origin of the often-cited **360 initial designs**. **Exactly 360 accepted CSV rows is *not* guaranteed**: forbidden substitutions, sequon changes, invalid mutation counts or duplicate sequences could reduce the count. The actual `01_all_model_proposals.csv` row count should be reported from that file rather than assumed.

### 5.3 Fast preselection: up to 8 per seed, or 40 total

For each seed, the script sorts valid proposals by masked log-odds, **prefers different newly mutated framework positions over alternate amino acids at the same position**, and takes **8 proposals per seed** (`--preselect-per-seed 8`). This yields **up to 40** sequences, written to `02_initially_ranked.csv`.

For these sequences the script also evaluates AbLang2's `confidence` mode, and subtracts the confidence score of the corresponding **unchanged seed**, yielding `confidence_delta_vs_seed`. **Importantly, this confidence number is recorded as an additional diagnostic; it does not drive the 8-per-seed preselection**. The primary selection at this step is masked log-odds with positional diversity.

### 5.4 Expensive rescoring: up to 6 per seed, or 30 finalists

From the preselected list, the script retains the **six highest masked-log-odds rows per seed** (`--pll-per-seed 6`) and computes paired AbLang2's **full masked pseudo-log-likelihood (PLL)**. This evaluates sequence plausibility across the complete paired variable-domain context and is more costly than one-position scoring.

It also computes each unchanged seed's PLL and records:

```text
pll_delta_vs_seed = PLL(full mutated VH/VL) − PLL(unchanged seed VH/VL)
```

These **up to 30** finalists, with absolute PLL and their seed-relative delta, are written to `03_pll_finalists.csv` and globally sorted by **PLL delta (descending)**, with masked log-odds as the tie-breaker.

A **positive delta** means a sequence receives higher AbLang2 plausibility than its seed; a **negative delta** means it scores lower. **The algorithm does not require a positive delta to be selected**: the downstream shortlist prioritizes a finite number of diverse candidates from distinct seeds. For example, in the completed screening metadata, `ML_20` has `pll_delta_vs_seed = −0.008674` despite its high Boltz-2 ipSAEmin. The model-based naturalness and structural-interface scores are distinct objectives.

### 5.5 Diversity-aware selection of 21 ML designs

The final selector (`select_diverse_ml_rows`) uses a **seed-balanced, round-robin-style greedy strategy**:

- Consider the five scan seeds; **prefer seeds with fewer previously selected candidates**.
- Within each seed, favor an **as-yet unused new mutation position** before repeating the same position with another amino acid.
- Among the eligible least-represented seeds, choose the candidate with highest **PLL delta vs its seed**, breaking ties by masked log-odds.
- Reject duplicate full sequences and repeat until **21** candidates are chosen.

The resulting candidate names, **`ML_01` through `ML_21`**, indicate **selection order**, **not performance rank**. The final 21 include **five V1-derived designs, four V2, four V3, four V5 and four SPR4** (verified from the completed Boltz-2 results CSV). No AbLang2 design receives more than **one additional substitution** compared with its seed.

### 5.6 Assemble the fixed 30-sequence screening shortlist

The code concatenates:

| Component | Names | Count | Evidence status |
|---|---|---:|---|
| Measured seeds | `V1`, `V2`, `V3`, `V4`, `V5` | **5** | Original combinations measured in AlphaSeq; challenge-specific validation pending |
| Rational designs | `B1`, `B2`, `B3`, `B4` | **4** | New untested combinations |
| AbLang2 variants | `ML_01` … `ML_21` | **21** | New untested single-framework extensions |
| **Total** | | **30** | Computational shortlist, **not** final lab submission |

The code asserts that there are **exactly 30** rows and no duplicate complete amino-acid sequences. The final filename is `04_boltz2_shortlist_30.csv`. **WT, SPR2, STABILITY2 and SPR4 are *not* among the 30**. WT is evaluated separately as a baseline; other excluded references stay in `00_reference_sequences.csv`.

## 6. Full provenance of the 21 generated ML variants

This table was **reconstructed from the uploaded completed 30-candidate Boltz-2 results CSV**, which includes the original shortlist's sequence metadata and thus identifies each selected seed and new mutation. Each entry adds precisely **one** new substitution to its named seed (seed substitutions are listed in §3.1 or §3.2).

<!-- BEGIN AUTO-GENERATED ML PROVENANCE TABLE -->
| Variant | Starting seed | One new AbLang2 substitution | Total mutations | Final full-scFv mutation set |
|---|---|---|---:|---|
| `ML_01` | `V2` | `A72R` | 6 | `Y52S,K65Q,A72R,S75D,K76S,M107F` |
| `ML_02` | `V3` | `A72R` | 6 | `N55D,A72R,V93T,D102Y,M107F,D108Q` |
| `ML_03` | `SPR4` | `A72R` | 5 | `A72R,S75D,R87T,S97A,D102Y` |
| `ML_04` | `V1` | `A72R` | 7 | `T54S,G66D,A72R,T74I,Q82E,D102A,M107F` |
| `ML_05` | `V5` | `A72R` | 6 | `A72R,G100L,G101I,D102G,A106D,M107F` |
| `ML_06` | `V2` | `N84S` | 6 | `Y52S,K65Q,S75D,K76S,N84S,M107F` |
| `ML_07` | `V3` | `N84D` | 6 | `N55D,N84D,V93T,D102Y,M107F,D108Q` |
| `ML_08` | `V5` | `A88T` | 6 | `A88T,G100L,G101I,D102G,A106D,M107F` |
| `ML_09` | `V1` | `A88V` | 7 | `T54S,G66D,T74I,Q82E,A88V,D102A,M107F` |
| `ML_10` | `SPR4` | `N84D` | 5 | `S75D,N84D,R87T,S97A,D102Y` |
| `ML_11` | `V1` | `N84S` | 7 | `T54S,G66D,T74I,Q82E,N84S,D102A,M107F` |
| `ML_12` | `V2` | `A88V` | 6 | `Y52S,K65Q,S75D,K76S,A88V,M107F` |
| `ML_13` | `V5` | `N84D` | 6 | `N84D,G100L,G101I,D102G,A106D,M107F` |
| `ML_14` | `V3` | `S211T` | 6 | `N55D,V93T,D102Y,M107F,D108Q,S211T` |
| `ML_15` | `SPR4` | `S211T` | 5 | `S75D,R87T,S97A,D102Y,S211T` |
| `ML_16` | `V2` | `S211T` | 6 | `Y52S,K65Q,S75D,K76S,M107F,S211T` |
| `ML_17` | `V1` | `S211T` | 7 | `T54S,G66D,T74I,Q82E,D102A,M107F,S211T` |
| `ML_18` | `V5` | `S211T` | 6 | `G100L,G101I,D102G,A106D,M107F,S211T` |
| `ML_19` | `V3` | `A23V` | 6 | `A23V,N55D,V93T,D102Y,M107F,D108Q` |
| `ML_20` | `SPR4` | `A23V` | 5 | `A23V,S75D,R87T,S97A,D102Y` |
| `ML_21` | `V1` | `A23V` | 7 | `A23V,T54S,G66D,T74I,Q82E,D102A,M107F` |

<!-- END AUTO-GENERATED ML PROVENANCE TABLE -->

## 7. Structural screening: Boltz-2 + ipSAE (after sequence generation)

The shortlist is an **input to a separate structure-screening stage**, not an AbLang2 output score. Our scripts `eval.py` and `modal_eval.py` execute a **self-installed Boltz-2 model on a Modal cloud H100** (not an external Boltz API), predict candidate scFv–HER2 complexes and score their predicted interfaces.

**Exact default settings in the version used here:**

| Parameter | Value | Reason |
|---|---|---|
| Complex chains | A: **242-aa scFv**, B: **630-aa HER2 ECD**, UniProt P04626 residues 23–652 | Evaluate the challenge construct against its required target segment |
| AbLang2 stage | Paired VH/VL only, no linker | Correct model input representation |
| Boltz-2 MSA | `empty` for scFv **and** HER2 in these runs | Speed and consistency across the entire WT/variant batch; reduced evolutionary information |
| Boltz-2 recycling steps | **3** | Modest inference budget |
| Sampling steps | **200** | Default inference setting used in scripts |
| Diffusion samples | **1 per variant** | Conserve GPU compute; leaves model sampling uncertainty underexplored |
| PAE cutoff for ipSAE | **15 Å** | Inter-chain PAE threshold implemented in evaluation script |
| Ranking key | **`ipsae_min` descending** | Conservative of two directed ipSAE metrics |
| WT handling | **Separate prediction and score** | Consistent comparison baseline; WT not eligible for mutated shortlist |

The custom evaluator loads the predicted PAE matrix, partitions its **scFv→HER2** and **HER2→scFv** blocks, and calculates a directed ipSAE-type score from low-PAE residue pairs. For each directed block it computes per-residue weighted scores using a length-adaptive normalization and takes the maximal residue score. It records:

```text
ipsae_min = min(ipsae_ab_to_her2, ipsae_her2_to_ab)
ipsae_max = max(ipsae_ab_to_her2, ipsae_her2_to_ab)
```

This is the **project's specific implementation** in `eval.py`; for exact formulas consult `d0_for_n`, `directed_ipsae` and `calculate_ipsae`. `ipTM`, `complex_plddt` and confidence data are read from Boltz's confidence JSONs. The summary uses **model index 0** (rather than maximizing ipSAE over replicate samples, which would bias selection). Structure `.cif`, PAE `.npz` and confidence `.json` files are retained on Modal's persistent volume and can be downloaded for inspection.

### 7.1 Actual screen outcome at the time of documentation

The supplied `ranked_ipsae.csv` contains **30/30 status=`ok`** variant predictions and a separate WT `ranked_ipsae.csv` contains **1/1 status=`ok`**. Based on these specific single-sample, empty-MSA results:

| Candidate | Origin seed | New ML mutation (if any) | ipSAEmin | ipTM | Comparison to WT ipSAEmin |
|---|---|---|---:|---:|---|
| `ML_20` | SPR4 | `A23V` | **0.333988** | **0.775899** | Higher |
| `ML_16` | V2 | `S211T` | **0.278914** | **0.721449** | Higher |
| `ML_15` | SPR4 | `S211T` | **0.207640** | **0.710924** | Higher |
| `ML_06` | V2 | `N84S` | 0.195720 | 0.666769 | Higher |
| `ML_01` | V2 | `A72R` | 0.189093 | 0.669939 | Higher |
| `ML_11` | V1 | `N84S` | 0.187744 | 0.679613 | Higher |
| `ML_12` | V2 | `A88V` | 0.169067 | 0.652203 | Higher |
| `V2` | Experimentally measured | — | 0.134557 | 0.622143 | Higher |
| **WT** | Parent | — | **0.114525** | **0.631211** | Baseline |

**Eight of the 30** exceed WT's ipSAEmin in this run. **Seven of those eight** also exceed WT's ipTM; `V2` has ipTM slightly **below** WT despite its higher ipSAEmin. `ML_20` and `ML_15` are particularly informative because they share the **same SPR4 seed** and differ only in their *one added framework mutation* (`A23V` vs `S211T`). `ML_16` tests whether the VL substitution `S211T` changes predictions on a **different, experimentally measured V2 background**.

**Limitations:** These are *model confidence proxies*, not measured binding affinity, docking free energy or K_D. The predictions use empty MSAs and only one diffusion sample each. A score of zero for a candidate does **not** prove no binding; it means the particular scorer did not identify sufficient favorable inter-chain PAE evidence under its threshold and model prediction. Structural geometry, paratope orientation, interface residue identities, disulfide integrity, expression, solubility and aggregation still need inspection.

## 8. Files and reproducibility

The implemented pipeline writes these files in `ablang2_results/`:

| File | Purpose | Expected number of rows (default configuration) |
|---|---|---:|
| `00_reference_sequences.csv` | WT, SPR2, STABILITY2, SPR4, V1–V5, B1–B4 with full sequences and provenance | **13** |
| `01_all_model_proposals.csv` | Every valid single-framework AbLang2 proposal with mask log-odds/QC | **At most 360**; depends on filters |
| `02_initially_ranked.csv` | Top 8 per scan seed with score annotations | **Up to 40** |
| `03_pll_finalists.csv` | Top 6 per seed with full pseudo-PLL and seed deltas | **Up to 30** |
| `04_boltz2_shortlist_30.csv` | The 5 + 4 + 21 design panel for Boltz-2 | **30** |
| `manifest.json` | Configuration, allowed/protected sites, assumptions, hazard notes | One JSON file |

**Commands (using the documented defaults):**

```bash
# From the repository root; requires Python with torch, ablang2, pandas, numpy
python trastuzumab_ablang2_design.py --dry-run
python trastuzumab_ablang2_design.py --output-dir ablang2_results --device cpu

# Optional local input inspection
python eval.py --wt-only --prepare-only

# GPU predictions on Modal (incurs GPU usage costs)
modal run modal_eval.py --wt-only
modal run modal_eval.py --limit 30

# Re-score/retrieve EXISTING predictions (no GPU requested)
modal run modal_eval.py --wt-only --score-only
modal run modal_eval.py --limit 30 --score-only --download-variants
```

The actual CLI for the second stage may evolve; `python eval.py --help` and `modal run modal_eval.py --help` should be treated as the executable source of truth. GPU function timeout is **3600 seconds** in the documented Modal wrapper; **this is not a dollar-spending cap**. No unverified cost guarantee should be made.

**Requirements and provenance:**

- The generation script checks parent sequence length (**242**), domain lengths, masked-token indexing, non-overlapping search/protected positions and duplicates.
- `manifest.json` records the actual configured search rules and should be committed alongside the script and result CSVs when possible.
- **Record package versions, model versions and hashes of the 30 sequences** for durable reproducibility. The current design script uses the official pretrained `ablang2-paired` checkpoint but does not pin a model-weight checksum or every Python dependency.
- The review here reconstructs the implemented logic directly from `trastuzumab_ablang2_design.py` and the uploaded final `ranked_ipsae.csv`. The intermediate proposal/finalist CSVs were **not available for independent row-count or score reconstruction** when writing this document. Therefore, theoretical intermediate maxima are accurately labeled **up to** rather than falsely claiming observed exact counts.

## 9. Interpretation, novel contribution and next experiments

The **original contribution is not the discovery of the AlphaSeq or SPR mutations**. It is the **combination and constrained exploration strategy**:

- Start from **experimentally informed HER2 binders** and complementary published observations rather than an unconstrained 20^242 search.
- Preserve the most important known antigen-contacting and structural features while testing **one extra framework change** that may alter variable-domain packing or sequence plausibility without redesigning the paratope wholesale.
- Retain **five distinct seed backgrounds** and diversify newly mutated positions, instead of only selecting the top likelihood scores from one background.
- Use a **hierarchical screening funnel** (masked log-odds → optional confidence diagnostic → full PLL delta → balanced 21-sequence selection → Boltz-2 interface scoring) to conserve compute.
- Separate **empirical AlphaSeq evidence**, **literature evidence**, **rational cross-source combinations** and **entirely unmeasured AbLang2 extensions** in all filenames and result metadata.

**Recommended next checks before the final 15-sequence lab submission:** inspect predicted antibody orientation and the HER2 contact patch (especially ML_20/ML_16/ML_15 versus WT), confirm mutation identity/construct lengths, compare interface confidence across additional samples where affordable, screen basic developability liabilities, keep some seed/background diversity, and make sure no computational score is described as experimental K_D. The final selection should be **15 unique eligible sequences** with ≥5 parent-relative mutations, not simply the first 15 rows of the current ranking.

## Sources

1. **Challenge definition:** Adaptyv Bio, *Trastuzumab scFv optimisation on HER2*, Zurich Hackathon deck, 31 October 2026 (project-supplied PDF).
2. **Experimental complex:** [PDB 1N8Z](https://www.rcsb.org/structure/1N8Z); Cho et al., *Nature* (2003), [DOI 10.1038/nature01392](https://doi.org/10.1038/nature01392).
3. **AlphaSeq measurements:** [A-Alpha Bio open AlphaSeq dataset](https://huggingface.co/datasets/aalphabio/open-alphaseq) and [Agarwal et al., AlphaBind (2025)](https://doi.org/10.1080/19420862.2025.2534626).
4. **Affinity-related CDR-H3 context:** [Shanehsazzadeh et al., Absci-associated public design data](https://github.com/AbSciBio/unlocking-de-novo-antibody-design).
5. **Framework stability:** [*Tuning antibody stability and function by rational designs of framework mutations*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12269682/).
6. **Model:** [Official AbLang2 implementation](https://github.com/oxpig/AbLang2) and [Olsen et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/39460949/).

---

*No binding or expression measurements have yet been obtained for our new B1–B4 or ML_01–ML_21 combinations. Predictions are exploratory and require wet-lab confirmation.*
