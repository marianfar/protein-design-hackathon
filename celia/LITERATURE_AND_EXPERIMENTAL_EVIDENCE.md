# Literature and experimental evidence for trastuzumab–HER2 variant design

**Project:** Adaptyv Bio Zurich Hackathon — trastuzumab scFv affinity optimization  
**Prepared:** 31 October 2026  
**Scope:** Evidence reviewed to select mutation seeds and design rational combinations. This is a provenance record, **not** a claim that our newly generated variants have experimentally improved affinity.

## 1. Biological context and numbering

The challenge starts with a **242-residue** trastuzumab single-chain variable fragment (scFv): **VH (residues 1–120) – `(GGGGS)3` linker (121–135) – VL (136–242)**. All mutation labels in our scripts, tables and this report use **1-based positions in this complete scFv**, not Kabat or IMGT numbering. An `S97A` substitution means serine at position 97 of the supplied 242-aa parent is changed to alanine. Framework/CDR descriptions below are approximate, using the ranges implemented in `trastuzumab_ablang2_design.py`.

The organizer reports a **parent K_D of 0.82 nM**, measured by **its own BLI platform** against HER2; the organizer's test uses the HER2 ectodomain, residues **Thr23–Thr652**. These values are the challenge-specific comparator, **not** the parent K_D measured in the literature datasets [1]. The crystal structure **PDB 1N8Z** shows human HER2 in complex with trastuzumab Fab and informs the choice to preserve the main antigen-contact region [2].

> **Crucial comparison rule:** A lower K_D indicates stronger binding *within a comparable assay*. Published/AlphaSeq K_D values are **not directly comparable** with the organizer's 0.82 nM BLI measurement, because antigen constructs, assay formats, antibody formats, conditions and reference constructs can differ. In particular, the design code flags an engineered HER2-OPT construct in the AlphaSeq experiments; confirm the exact assay metadata before equating those values with binding to the challenge's native HER2 ectodomain.

## 2. Experimental AlphaSeq sequences selected as seeds

We selected **five already measured, multi-substitution trastuzumab-derived sequences** from A-Alpha Bio's open AlphaSeq resource [3], which accompanies the AlphaBind study [4]. The recorded data below are the values incorporated into `MEASURED_META` in our design script and exported in `00_reference_sequences.csv`. The mutations represent **complete measured variants**: the data do **not** establish that every constituent substitution is individually beneficial.

| Internal ID | AlphaSeq experiment | Mutations relative to challenge scFv | Variant K_D (nM) | Same-experiment comparator K_D (nM) | Relative improvement* |
|---|---|---|---:|---:|---:|
| **V1** | `YM_0990` (CDR + framework) | `T54S, G66D, T74I, Q82E, D102A, M107F` | 27.70 | 102.93 | **3.72×** |
| **V2** | `YM_0990` (CDR + framework) | `Y52S, K65Q, S75D, K76S, M107F` | 36.59 | 102.93 | **2.81×** |
| **V3** | `YM_0990` (CDR + framework) | `N55D, V93T, D102Y, M107F, D108Q` | 38.43 | 102.93 | **2.68×** |
| **V4** | `YM_0990` (CDR + framework) | `N55D, E89D, V93T, D102Y, M107F` | 43.38 | 102.93 | **2.37×** |
| **V5** | `YM_0989` (CDR-H3 optimization) | `G100L, G101I, D102G, A106D, M107F` | 33.40 | 74.77 | **2.24×** |

\*Calculated as `(K_D of comparator)/(K_D of variant)`, within the *same* AlphaSeq experiment. The two comparator measurements differ between `YM_0990` and `YM_0989` and must not be conflated. These exact numbers and mappings are **transcribed from our project's reference CSV/code**, not independently re-extracted row-by-row from the live AlphaSeq dataset during preparation of this note. For a formal publication, retain the original experiment and row identifiers, assay uncertainties and experimental metadata.

**Why this was useful:** These are experimentally measured, HER2-related sequence combinations with **5–6 mutations**, directly meeting the hackathon's minimum of five substitutions. They provide evidence-guided alternatives to arbitrary random mutagenesis. V1, V2, V3 and V5 were also used as starting points for one further **framework-only** AbLang2 substitution; **V4 was included in the fixed 30-candidate shortlist but was not scanned**.

**Important distinction:** `V2` has experimental AlphaSeq evidence, whereas an ML variant derived from it (e.g., `ML_16 = V2 + S211T`) does **not** have experimental evidence for the *new full combination*.

## 3. Published HCDR3 affinity-associated substitutions: S97A + D102Y

The Absci-associated study **“Unlocking de novo antibody design with generative artificial intelligence”** provides a public dataset with SPR binding measurements for human HER2-binding antibodies and designs around trastuzumab's CDR-H3 [5]. The study's data release includes `spr-controls.csv` and designed-binder tables, alongside the reference trastuzumab variable-domain sequences.

Our design file encodes the following **two-substitution control**:

| Design label | Mutations | Region | K_D recorded in our script | Comparator K_D recorded in our script | Fold difference |
|---|---|---|---:|---:|---:|
| `SPR2` | `S97A, D102Y` | VH CDR-H3 | 0.56 nM | 1.94 nM | 3.46× |

This **0.56 / 1.94 nM pair comes from the evidence mapping recorded in `trastuzumab_ablang2_design.py`**, attributed there to the Absci SPR panel; the relevant individual source data row should be cross-checked when turning this repository note into a publication-grade result. Do **not** reinterpret the 1.94 nM comparator as the organizer's 0.82 nM BLI parent.

Separately, patent literature collates **D102Y** as a trastuzumab heavy-chain substitution associated with an approximately **3.1×** binding improvement; this is a **secondary patent summary of earlier work**, not a new measurement by our project [6]. The reported effect of isolated `D102Y` is **not** the same result as the recorded two-mutation `S97A/D102Y` SPR construct.

**Design consequence:** `SPR2` was retained as an **affinity-related, two-mutation reference control**, but does **not** qualify for the hackathon's five-substitution minimum. We incorporated both substitutions into an additional, untested **four-mutation composite seed** `SPR4` (described below). We did **not** claim that combining published mutations preserves or multiplies their individual binding gains.

## 4. Framework stability evidence: S85N + R87T

The experimental study **“Tuning antibody stability and function by rational designs of framework mutations”** used a structure-informed framework-design workflow on trastuzumab and tested several double mutations [7]. Two important outcomes:

- **`S85N + R87T`** (VH framework): improved thermal stability relative to wild type while maintaining HER2 binding and downstream function in the assays reported. **It was not demonstrated to improve HER2 affinity.**
- **`R50S + R59N`**: increased thermostability but compromised HER2 binding and functional activity. It is a cautionary example of why framework changes require validation beyond stability.

The `S85N + R87T` pair changes scFv residues 85–87 to **`N-L-T`**, an `N-X-S/T` **potential N-linked glycosylation sequon**. The publication explicitly discusses the glycosylation caveat. Our design therefore:

1. Keeps `STABILITY2 = S85N,R87T` as a **reference-only** sequence with an explicit glycosylation-risk label.
2. Reuses **`R87T` alone** in the constructed `SPR4` seed while omitting `S85N`, avoiding that particular newly created `N-L-T` sequon.
3. Does **not** claim that `R87T` alone reproduces the pair's stability benefit. That was **not** what the cited experiment tested.

## 5. Additional literature-reported affinity mutations considered as context

A HER2 antibody patent compilation lists the following **trastuzumab** substitutions and approximate reported improvements [6]. These are useful prior-art leads **but were not directly introduced as fixed designs by the final implemented sequence-generation script** (except that `D102Y` occurs in some selected seeds through the evidence streams above).

| Listed substitution | Reported improvement | Caveat |
|---|---:|---|
| VH `D102W` | 3.2× | Patent notes an alternative `D98W` numbering convention; mapping requires care. |
| VH `D102Y` | 3.1× | Appears in our selected `V3`, `V4`, `SPR2` and `SPR4` lines. |
| VH `D102K` | 2.3× | Not included in the final implemented shortlist as a standalone new substitution. |
| VH `D102T` | 2.2× | Same. |
| VH `N55K` | 2.0× | Not the same as the measured seed substitution `N55D`. |
| VH `N55T` | 1.9× | Not the same as `N55D`. |
| VL `H91F` | 2.1× | **VL-local numbering** in patent; not scFv position 91. |
| VL `D28R` | 1.9× | **VL-local numbering** in patent; not scFv position 28. |

The patent also lists **pertuzumab** substitutions; these are **not trastuzumab substitutions and were not transplanted into our design**. Patent tables summarize reported improvements under potentially different assays and formats: they should be used as leads, not as directly comparable measurements or independent replication.

## 6. Rational combinations assembled specifically for this project

The sequence-generation pipeline created the following **new composite sequences**, with no experimental measurements for these exact multi-mutant combinations:

| Internal name | Full mutation list | Why constructed | Included in 30? |
|---|---|---|---|
| `SPR4` | `S97A, D102Y, S75D, R87T` | Combine the SPR-associated CDR-H3 pair, an AlphaSeq-derived position (`S75D`) and one residue from the published stability pair (`R87T`). | **No**, only 4 mutations; used as AbLang2 seed. |
| `B1` | `S97A, D102Y, S75D, R87T, Y52S` | Add AlphaSeq-derived `Y52S` to reach 5 changes. | **Yes** |
| `B2` | `S97A, D102Y, S75D, R87T, G101A` | Explore a CDR-H3 change at 101 on the composite background. | **Yes** |
| `B3` | `S97A, D102Y, S75D, R87T, G100L` | Combine the composite seed with `G100L` seen in V5. | **Yes** |
| `B4` | `S97A, D102Y, S75D, R87T, T54D` | Test an alternative HCDR2 substitution on the composite background. | **Yes** |

`G101A` in B2 and `T54D` in B4 are **rational hypotheses, not the measured `G101I` and `T54S` substitutions** from V5 and V1 respectively. The origin of a residue in a tested seed **does not validate a different replacement at that position**. All B1–B4 combinations were *untested at generation time*.

## 7. Source-to-design provenance and what is actually demonstrated

| Evidence stream | Empirical evidence? | Used how? | What cannot be concluded |
|---|---|---|---|
| AlphaSeq `V1`–`V5` | **Yes, complete variants** in AlphaSeq assay | All five carried forward as fixed candidates; V1/V2/V3/V5 also AbLang2 seeds | Each constituent mutation individually improves binding; values equal challenge BLI K_D |
| Absci-associated `S97A/D102Y` | **SPR values recorded in project's provenance table**; see source-row verification note | `SPR2` reference, core of `SPR4` | New SPR4/B1–B4 combinations inherit the SPR2 K_D |
| Published `S85N/R87T` | **Yes, stability and functional characterization** | Reference; `R87T` adopted alone in composite design | `R87T` alone is affinity-enhancing or reproduces measured stability |
| Patent-listed single substitutions | **Literature/prior-art summary** | Contextual search space; not directly substituted into script as an independent set | All patent fold values are transferable or independently confirmed by our assay |
| `SPR4`, `B1`–`B4` | **No** | New rational combinations | Improved experimental affinity or stability |
| `ML_01`–`ML_21` | **No** | AbLang2 proposed one additional permitted framework change | Improved affinity, even if predicted confidence is higher |

Our own Boltz-2/ipSAE evaluation is **structural confidence-based computational screening**, not a binding assay. In the current uploaded results, **8/30** candidates have `ipSAEmin` greater than WT, but this is **not** the same as eight experimentally improved binders. See `SEQUENCE_GENERATION_PIPELINE.md` for exact design, selection, scoring and provenance details.

## References and source datasets

1. **Adaptyv Bio (2026).** *Trastuzumab scFv optimisation on HER2*, Zurich hackathon challenge deck, 31 October 2026 (project-provided PDF; parent sequence, organizer BLI K_D, rules and HER2 construct).
2. **Cho, H.-S. et al. (2003).** *Structure of the extracellular region of HER2 alone and in complex with the Herceptin Fab.* Nature 421:756–760. [DOI: 10.1038/nature01392](https://doi.org/10.1038/nature01392); [PDB 1N8Z](https://www.rcsb.org/structure/1N8Z).
3. **A-Alpha Bio (2025).** *Open AlphaSeq affinity datasets*, including `YM_0989` and `YM_0990`. [Dataset and experiment glossary](https://huggingface.co/datasets/aalphabio/open-alphaseq).
4. **Agarwal, A. A. et al. (2025).** *AlphaBind, a domain-specific model to predict and optimize antibody–antigen binding affinity.* mAbs 17(1), 2534626. [DOI: 10.1080/19420862.2025.2534626](https://doi.org/10.1080/19420862.2025.2534626); [PubMed](https://pubmed.ncbi.nlm.nih.gov/40693434/).
5. **Shanehsazzadeh, A. et al. (2023 preprint).** *Unlocking de novo antibody design with generative artificial intelligence.* [DOI: 10.1101/2023.01.08.523187](https://doi.org/10.1101/2023.01.08.523187); [public Absci-associated data](https://github.com/AbSciBio/unlocking-de-novo-antibody-design).
6. **Patent prior-art summary:** [US20210260210A1, Table 4, trastuzumab mutations](https://patents.google.com/patent/US20210260210A1/en) (also reproduced in related HER2 biparatopic patent families). Note: a secondary compilation of mutation effects.
7. **Tuning antibody stability and function by rational designs of framework mutations** (2025). [Full text, PMCID PMC12269682](https://pmc.ncbi.nlm.nih.gov/articles/PMC12269682/).
8. **Olsen, T. H., Moal, I. H. & Deane, C. M.** *Addressing the antibody germline bias and its effect on language models for improved antibody design.* [AbLang2 official repository](https://github.com/oxpig/AbLang2); [PubMed](https://pubmed.ncbi.nlm.nih.gov/39460949/).

### Reproducibility note

The authoritative **implementation-side** provenance is `trastuzumab_ablang2_design.py` plus `ablang2_results/00_reference_sequences.csv`. The current manuscript-style summary deliberately separates (a) source-supported findings, (b) experimental numbers transcribed into our project files but not independently row-checked here, and (c) rational/ML designs that have **not** been experimentally measured.
