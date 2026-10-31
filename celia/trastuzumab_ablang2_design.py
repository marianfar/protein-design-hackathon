#!/usr/bin/env python3
"""Constrained AbLang2 design of trastuzumab scFv variants for a HER2 challenge.

Requirements (Python 3.10+):
    python -m pip install torch ablang2 pandas numpy

Quick start:
    python trastuzumab_ablang2_design.py --dry-run
    python trastuzumab_ablang2_design.py --output-dir ablang2_results

Run on an Apple Silicon Mac using --device cpu initially. The first non-dry run
downloads the official AbLang2 model weights. Optionally set --device mps if
your particular PyTorch/AbLang2 installation works with Metal acceleration.

Inputs: exact 242-residue hackathon parent; five AlphaSeq-measured designs;
the Absci SPR HCDR3 affinity pair; and a literature framework stability pair.
Positions are 1-based within the full 242-residue VH-linker-VL scFv.

Outputs under --output-dir:
    00_reference_sequences.csv   Parent, measured variants, rational combinations
    01_all_model_proposals.csv  Every valid one-position AbLang2 proposal
    02_initially_ranked.csv     Top proposals per seed, rapid score included
    03_pll_finalists.csv        Full masked pseudo-log-likelihood of finalists
    04_boltz2_shortlist_30.csv  5 measured + 4 rational + 21 model variants
    manifest.json              Configuration and risk warnings

Important: AbLang2 scores sequence plausibility, NOT HER2 affinity. Neither
the generated proposals nor rationally combined variants are validated binders.
The 30-variant shortlist is NOT the final 15-sequence submission. Run Boltz-2
complex predictions and interface analyses (including ipSAE) on the shortlist,
then select the final 15 using structure, experimental evidence and diversity.
Use WT and SPR2 from 00_reference_sequences.csv as additional controls; neither
qualifies as a five-substitution hackathon submission.
The original AlphaSeq data used an engineered HER2-OPT target and a different
assay from the Adaptyv challenge; absolute KD values are not interchangeable.

Primary data:
    https://huggingface.co/datasets/aalphabio/open-alphaseq
    https://github.com/AbSciBio/unlocking-de-novo-antibody-design
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12269682/
AbLang2 authors' implementation:
    https://github.com/oxpig/AbLang2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEW"
    "VARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVY"
    "YCSRWGGDGFYAMDYWGQGTLVTVSS"
)
VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKL"
    "LIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHY"
    "TTPPTFGQGTKVEIK"
)
LINKER = "GGGGS" * 3
PARENT = VH + LINKER + VL
AA = "ACDEFGHIKLMNPQRSTVWY"

# Residues not freely redesigned: primary HER2 contacts observed in 1N8Z.
# Experimentally tested interface substitutions (e.g., Y52S, D102Y) are
# excluded from this hard-lock set, but are still NOT freely masked.
CONTACT_LOCK = frozenset({
    33, 50, 57, 58, 59, 99, 103, 104, 105,
    163, 164, 165, 166, 167, 184, 185, 188, 201,
    226, 227, 228, 229,
})
DISULFIDE_LOCK = frozenset({22, 96, 158, 223})
LINKER_LOCK = frozenset(range(121, 136))

# Exact experimentally guided positions. These are not model-discovery sites.
# An original mutation in a starting seed always remains unchanged.
EVIDENCE_ONLY_POSITIONS = frozenset({
    52, 54, 55, 65, 66, 74, 75, 76, 82, 85, 87, 89, 93,
    97, 100, 101, 102, 106, 107, 108,
})

CDR_POSITIONS = frozenset(
    list(range(26, 34)) + list(range(51, 59)) + list(range(97, 110))
    + list(range(162, 168)) + list(range(185, 188))
    + list(range(224, 233))
)

# First-round framework exploration whitelist from 1N8Z: approximate relative
# solvent accessibility >=25%, distance to HER2 >=10 A, and distance to the
# opposite variable domain >=8 A, with additional conservative exclusions.
ALLOW_VH = (7, 13, 17, 19, 23, 63, 69, 71, 72, 84, 88)
ALLOW_VL = (142, 144, 145, 147, 150, 152, 153, 155,
            195, 198, 211, 212, 216)
ALLOWED_POSITIONS = tuple(ALLOW_VH + ALLOW_VL)

# Exclude residues especially likely to create liabilities when introduced at
# a solvent-exposed framework site. This is a conservative proposal filter,
# not a statement that these amino acids are universally undesirable.
OMIT_NEW_AA = frozenset("CPGW")

MEASURED_MUTATIONS = {
    "V1": ("T54S", "G66D", "T74I", "Q82E", "D102A", "M107F"),
    "V2": ("Y52S", "K65Q", "S75D", "K76S", "M107F"),
    "V3": ("N55D", "V93T", "D102Y", "M107F", "D108Q"),
    "V4": ("N55D", "E89D", "V93T", "D102Y", "M107F"),
    "V5": ("G100L", "G101I", "D102G", "A106D", "M107F"),
}

MEASURED_META = {
    "V1": ("YM_0990", 27.70, 102.93),
    "V2": ("YM_0990", 36.59, 102.93),
    "V3": ("YM_0990", 38.43, 102.93),
    "V4": ("YM_0990", 43.38, 102.93),
    "V5": ("YM_0989", 33.40, 74.77),
}

SPR2 = ("S97A", "D102Y")  # 0.56 nM versus 1.94 nM, Absci SPR panel
STABILITY2 = ("S85N", "R87T")  # N-L-T N-glycosylation sequon: reference only
SPR4 = ("S97A", "D102Y", "S75D", "R87T")  # untested 4-mutant

# Explicit combinations proposed in the discussion. These are untested.
RATIONAL_MUTATIONS = {
    "B1": ("S97A", "D102Y", "S75D", "R87T", "Y52S"),
    "B2": ("S97A", "D102Y", "S75D", "R87T", "G101A"),
    "B3": ("S97A", "D102Y", "S75D", "R87T", "G100L"),
    "B4": ("S97A", "D102Y", "S75D", "R87T", "T54D"),
}

SCAN_SEEDS = ("V1", "V2", "V3", "V5", "SPR4")


def mutations_of(sequence: str) -> list[str]:
    """Return substitutions relative to the challenge parent (1-based)."""
    if len(sequence) != len(PARENT):
        raise ValueError("Candidate has an incorrect number of residues")
    return [f"{old}{i}{new}" for i, (old, new)
            in enumerate(zip(PARENT, sequence), 1) if old != new]


def make_sequence(mutations: tuple[str, ...]) -> str:
    seq = list(PARENT)
    for mut in mutations:
        pos = int(mut[1:-1])
        original, new = mut[0], mut[-1]
        if not (1 <= pos <= len(seq)):
            raise ValueError(f"Invalid position {mut}")
        if seq[pos - 1] != original:
            raise ValueError(f"Mutation {mut} disagrees with parent sequence")
        if new not in AA:
            raise ValueError(f"Invalid substitution: {mut}")
        seq[pos - 1] = new
    return "".join(seq)


def split_for_ablang2(sequence: str) -> list[str]:
    """AbLang2 sees VH and VL as a pair; the linker is omitted."""
    return [sequence[:120], sequence[135:]]


def n_glyc_sequons(sequence: str) -> set[int]:
    return {i + 1 for i in range(len(sequence) - 2)
            if sequence[i] == "N" and sequence[i + 1] != "P"
            and sequence[i + 2] in "ST"}


def validate_candidate(
    sequence: str,
    source_seed: str | None = None,
    added_position: int | None = None,
    max_total_mutations: int = 7,
) -> tuple[bool, str]:
    if len(sequence) != 242 or any(a not in AA for a in sequence):
        return False, "Length or amino-acid alphabet error"
    if sequence[120:135] != LINKER:
        return False, "Linker changed"
    if any(sequence[pos - 1] != PARENT[pos - 1]
           for pos in CONTACT_LOCK | DISULFIDE_LOCK | LINKER_LOCK):
        return False, "Protected residue changed"
    observed = mutations_of(sequence)
    if not (5 <= len(observed) <= max_total_mutations):
        return False, "Mutation count outside permitted design range"
    if source_seed is not None:
        # The model may ADD a mutation but may not revert/change any seed site.
        prior_changes = mutations_of(source_seed)
        if any(sequence[int(mut[1:-1]) - 1] != mut[-1]
               for mut in prior_changes):
            return False, "Existing seed substitution changed"
        if added_position not in ALLOWED_POSITIONS:
            return False, "Non-whitelisted AbLang2 mutation position"
        if sequence[added_position - 1] in OMIT_NEW_AA:
            return False, "Introduced a disallowed framework amino acid"
        if n_glyc_sequons(sequence) - n_glyc_sequons(source_seed):
            return False, "Introduced an N-X-S/T glycosylation sequon"
    if sequence.count("C") != PARENT.count("C"):
        return False, "Cysteine count changed"
    if n_glyc_sequons(sequence) - n_glyc_sequons(PARENT):
        return False, "Introduced an N-X-S/T sequon relative to parent"
    return True, "pass"


def create_reference_table() -> pd.DataFrame:
    entries: list[dict] = []

    def add(name: str, sequence: str, kind: str, dataset: str = "",
            kd: float | None = None, kd_parent: float | None = None) -> None:
        muts = mutations_of(sequence)
        if kind not in {"reference_only", "reference_only_glycosylation_risk"}:
            valid, why = validate_candidate(sequence, max_total_mutations=7)
            if not valid:
                raise AssertionError(f"Reference {name} invalid: {why}")
        entries.append({
            "name": name, "evidence_type": kind,
            "mutation_count": len(muts), "mutations": ",".join(muts),
            "dataset": dataset,
            "study_KD_nM": kd, "same_study_parent_KD_nM": kd_parent,
            "study_fold_improvement": (kd_parent / kd if kd and kd_parent else None),
            "sequence": sequence,
        })

    add("WT", PARENT, "reference_only")
    add("SPR2", make_sequence(SPR2), "reference_only", "Absci_SPR", .56, 1.94)
    add("STABILITY2", make_sequence(STABILITY2), "reference_only_glycosylation_risk")
    add("SPR4", make_sequence(SPR4), "reference_only")
    for name, mutations in MEASURED_MUTATIONS.items():
        assay, kd, kd_parent = MEASURED_META[name]
        add(name, make_sequence(mutations), "experimentally_measured", assay,
            kd, kd_parent)
    for name, mutations in RATIONAL_MUTATIONS.items():
        add(name, make_sequence(mutations), "rational_untested")
    return pd.DataFrame(entries)


def mask_for_position(sequence: str, pos: int) -> tuple[list[str], int]:
    """Return VH/VL with one '*' and the AbLang2 token index.

    AbLang2 tokenizes as '<VH>|<VL>'. VH residue 1 occupies token index 1;
    VL residue 1 occupies len(VH)+4, or index 124 for this construct.
    """
    vh, vl = split_for_ablang2(sequence)
    if 1 <= pos <= 120:
        offset = pos - 1
        return [vh[:offset] + "*" + vh[offset + 1:], vl], pos
    if 136 <= pos <= 242:
        offset = pos - 136
        return [vh, vl[:offset] + "*" + vl[offset + 1:]], len(VH) + 4 + offset
    raise ValueError("Cannot mask an scFv linker position")


def scan_masked_sites(model, seed_name: str, seed: str,
                      per_position: int, max_total: int) -> list[dict]:
    import torch

    proposals = []
    for pos in ALLOWED_POSITIONS:
        masked_pair, token_pos = mask_for_position(seed, pos)
        tokens = model.tokenizer([masked_pair], pad=True,
                                 w_extra_tkns=True, device=model.used_device)
        if int(tokens[0, token_pos]) != int(model.tokenizer.mask_token):
            raise AssertionError(f"Masked token index is wrong at scFv {pos}")

        with torch.inference_mode():
            logits = model.AbLang(tokens)[0, token_pos]
            log_p = torch.log_softmax(logits, dim=-1)

        original = seed[pos - 1]
        orig_token = model.tokenizer.aa_to_token[original]
        choices = []
        for new_aa in AA:
            if new_aa == original or new_aa in OMIT_NEW_AA:
                continue
            new_token = model.tokenizer.aa_to_token[new_aa]
            log_odds = float((log_p[new_token] - log_p[orig_token]).item())
            choices.append((new_aa, log_odds))
        choices.sort(key=lambda item: item[1], reverse=True)

        for aa, log_odds in choices[:per_position]:
            candidate = seed[:pos - 1] + aa + seed[pos:]
            valid, reason = validate_candidate(candidate, seed, pos, max_total)
            if not valid:
                continue
            proposals.append({
                "seed": seed_name,
                "new_mutation": f"{original}{pos}{aa}",
                "new_position": pos,
                "masked_log_odds": log_odds,
                "mutation_count": len(mutations_of(candidate)),
                "mutations": ",".join(mutations_of(candidate)),
                "evidence_type": "AbLang2_unvalidated",
                "QC": reason,
                "sequence": candidate,
            })
    return proposals


def score_mode(model, sequences: list[str], mode: str,
               batch_size: int = 16) -> list[float]:
    if not sequences:
        return []
    outputs = []
    for first in range(0, len(sequences), batch_size):
        batch = [split_for_ablang2(s)
                 for s in sequences[first:first + batch_size]]
        values = model(batch, mode=mode, align=False,
                       batch_size=batch_size)
        # The authors' API returns numpy arrays for sequence scores.
        outputs.extend(float(value) for value in values)
    return outputs


def preselect_distinct_sites(group: pd.DataFrame, total: int) -> pd.DataFrame:
    """Prefer different framework positions before alternate AAs at one site."""
    ordered = group.sort_values("masked_log_odds", ascending=False)
    distinct = ordered.drop_duplicates(subset="new_position").head(total)
    if len(distinct) == total:
        return distinct
    remaining = ordered.drop(index=distinct.index).head(total - len(distinct))
    return pd.concat([distinct, remaining], ignore_index=False)


def select_diverse_ml_rows(df: pd.DataFrame, total: int) -> pd.DataFrame:
    """Round-robin selection across seeds; prefer distinct added positions."""
    if df.empty:
        return df
    ordered = df.sort_values(["pll_delta_vs_seed", "masked_log_odds"],
                             ascending=[False, False])
    selections: list[pd.Series] = []
    selected_sequences: set[str] = set()
    used_positions: dict[str, set[int]] = {seed: set() for seed in SCAN_SEEDS}
    selected_counts: dict[str, int] = {seed: 0 for seed in SCAN_SEEDS}

    while len(selections) < total:
        candidates: list[tuple[str, pd.Series]] = []
        for seed in SCAN_SEEDS:
            available = ordered[(ordered.seed == seed)
                                & (~ordered.sequence.isin(selected_sequences))]
            if available.empty:
                continue
            different_site = available[
                ~available.new_position.isin(used_positions[seed])
            ]
            row = (different_site if not different_site.empty else available).iloc[0]
            candidates.append((seed, row))
        if not candidates:
            break
        fewest_from_seed = min(selected_counts[seed] for seed, _ in candidates)
        balanced = [(seed, row) for seed, row in candidates
                    if selected_counts[seed] == fewest_from_seed]
        seed, row = max(balanced, key=lambda item:
                        (item[1].pll_delta_vs_seed, item[1].masked_log_odds))
        selections.append(row)
        selected_sequences.add(row.sequence)
        used_positions[seed].add(int(row.new_position))
        selected_counts[seed] += 1

    return pd.DataFrame(selections).head(total)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("ablang2_results"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate all seeds/position sets without downloading a model")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-aa-per-position", type=int, default=3)
    parser.add_argument("--preselect-per-seed", type=int, default=8)
    parser.add_argument("--pll-per-seed", type=int, default=6)
    parser.add_argument("--max-total-mutations", type=int, default=7)
    parser.add_argument("--shortlist-size", type=int, default=30,
                        help="Total variants for Boltz-2 (9 fixed designs + AbLang2 designs)")
    args = parser.parse_args()

    if len(VH) != 120 or len(VL) != 107 or len(PARENT) != 242:
        raise AssertionError("Parent domains do not match the hackathon sequence")
    if set(ALLOWED_POSITIONS) & (CONTACT_LOCK | DISULFIDE_LOCK | LINKER_LOCK
                                 | EVIDENCE_ONLY_POSITIONS | CDR_POSITIONS):
        raise AssertionError("The model mutation whitelist overlaps protected sites")
    if len(set(ALLOWED_POSITIONS)) != 24:
        raise AssertionError("Expected exactly 24 discovery positions")
    for pos in ALLOWED_POSITIONS:
        masked_pair, token_pos = mask_for_position(PARENT, pos)
        ablang_input = f"<{masked_pair[0]}>|<{masked_pair[1]}>"
        if ablang_input.count("*") != 1 or ablang_input[token_pos] != "*":
            raise AssertionError(f"Bad AbLang2 mask mapping for residue {pos}")
    if not (1 <= args.top_aa_per_position <= 16):
        parser.error("--top-aa-per-position must be between 1 and 16")
    if args.pll_per_seed > args.preselect_per_seed:
        parser.error("--pll-per-seed cannot exceed --preselect-per-seed")
    n_fixed_designs = len(MEASURED_MUTATIONS) + len(RATIONAL_MUTATIONS)
    n_model_variants = args.shortlist_size - n_fixed_designs
    if n_model_variants < 1:
        parser.error(f"--shortlist-size must exceed {n_fixed_designs}")
    if n_model_variants > len(SCAN_SEEDS) * args.pll_per_seed:
        parser.error("Not enough pseudo-PLL finalists for the requested shortlist: "
                     "increase --pll-per-seed and, if necessary, --preselect-per-seed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = create_reference_table()
    write_csv(reference, args.output_dir / "00_reference_sequences.csv")

    manifest = {
        "parent_length": len(PARENT),
        "coordinate_system": "1-based positions in 242-aa VH-linker-VL scFv",
        "VH_length": len(VH), "linker_length": len(LINKER), "VL_length": len(VL),
        "model": "ablang2-paired",
        "scan_seeds": SCAN_SEEDS,
        "scan_positions": ALLOWED_POSITIONS,
        "contact_lock": sorted(CONTACT_LOCK),
        "disulfide_lock": sorted(DISULFIDE_LOCK),
        "linker_lock": sorted(LINKER_LOCK),
        "evidence_only_positions": sorted(EVIDENCE_ONLY_POSITIONS),
        "omit_new_amino_acids": sorted(OMIT_NEW_AA),
        "max_total_mutations": args.max_total_mutations,
        "top_aa_per_position": args.top_aa_per_position,
        "preselect_per_seed": args.preselect_per_seed,
        "pll_per_seed": args.pll_per_seed,
        "boltz2_shortlist_size": args.shortlist_size,
        "boltz2_shortlist_measured": len(MEASURED_MUTATIONS),
        "boltz2_shortlist_rational": len(RATIONAL_MUTATIONS),
        "boltz2_shortlist_ablang2": n_model_variants,
        "notes": [
            "This is candidate generation and sequence naturalness scoring, not KD prediction.",
            "The 30-variant Boltz-2 shortlist is an intermediate output, not the final 15 submission.",
            "Run Boltz-2 complex prediction, calculate ipSAE and inspect interface contacts before selecting 15.",
            "WT and SPR2 are additional structural reference controls in 00_reference_sequences.csv.",
            "B1-B4 and all AbLang2 designs are untested combinations.",
            "S85N+R87T makes N-L-T at positions 85-87 (potential N-linked glycosylation); excluded from default designs.",
            "Study-specific SPR/AlphaSeq KD must not be compared numerically to Adaptyv KD.",
            "Review aggregation, charge patches, beta-strand packing, and HER2 pose before submission.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        print("Dry-run passed: seed numbering and all hard locks validated.")
        print("Scan positions:", ",".join(map(str, ALLOWED_POSITIONS)))
        print("Run again without --dry-run to load AbLang2 and generate proposals.")
        return

    try:
        import torch
        import ablang2
    except ImportError as exc:
        raise SystemExit(
            "Install requirements: python -m pip install torch ablang2 pandas numpy"
        ) from exc
    torch.set_num_threads(args.threads)
    print("Loading the authors' AbLang2 paired model...", flush=True)
    model = ablang2.pretrained(model_to_use="ablang2-paired",
                                random_init=False, ncpu=1, device=args.device)
    model.freeze()

    seed_sequences = {
        seed: make_sequence(SPR4 if seed == "SPR4" else MEASURED_MUTATIONS[seed])
        for seed in SCAN_SEEDS
    }

    all_proposals = []
    for name, seed in seed_sequences.items():
        print(f"Mask scanning {name} ({len(mutations_of(seed))} original mutations)...",
              flush=True)
        all_proposals.extend(scan_masked_sites(
            model, name, seed, args.top_aa_per_position,
            args.max_total_mutations
        ))

    if not all_proposals:
        raise RuntimeError("No designs passed QC; relax the model search only after review")
    proposals = pd.DataFrame(all_proposals)
    proposals = proposals.drop_duplicates(subset="sequence")
    proposals = proposals.sort_values("masked_log_odds", ascending=False)
    write_csv(proposals, args.output_dir / "01_all_model_proposals.csv")

    initial = pd.concat([
        preselect_distinct_sites(group, args.preselect_per_seed)
        for _, group in proposals.groupby("seed", sort=False)
    ], ignore_index=True)
    initial["confidence"] = score_mode(
        model, initial.sequence.tolist(), "confidence", batch_size=16)
    seed_confidence = dict(zip(
        seed_sequences,
        score_mode(model, list(seed_sequences.values()), "confidence", batch_size=8)
    ))
    initial["confidence_delta_vs_seed"] = [
        confidence - seed_confidence[seed]
        for seed, confidence in zip(initial.seed, initial.confidence)
    ]
    initial = initial.sort_values(
        ["seed", "masked_log_odds"], ascending=[True, False])
    write_csv(initial, args.output_dir / "02_initially_ranked.csv")

    # Full masked pseudo-log-likelihood is expensive; run only on the top few
    # alternatives per seed. Do not rank by the unmasked confidence alone.
    finalists = initial.groupby("seed", group_keys=False).head(args.pll_per_seed).copy()
    print(f"Computing full pseudo-log-likelihood for {len(finalists)} finalists...",
          flush=True)
    finalists["pseudo_log_likelihood"] = score_mode(
        model, finalists.sequence.tolist(), "pseudo_log_likelihood", batch_size=1)
    seed_pll = dict(zip(
        seed_sequences,
        score_mode(model, list(seed_sequences.values()),
                   "pseudo_log_likelihood", batch_size=1)
    ))
    finalists["seed_pseudo_log_likelihood"] = finalists.seed.map(seed_pll)
    finalists["pll_delta_vs_seed"] = (
        finalists.pseudo_log_likelihood - finalists.seed_pseudo_log_likelihood
    )
    finalists = finalists.sort_values(
        ["pll_delta_vs_seed", "masked_log_odds"], ascending=False)
    write_csv(finalists, args.output_dir / "03_pll_finalists.csv")

    selected = select_diverse_ml_rows(finalists, n_model_variants)
    if len(selected) != n_model_variants:
        raise RuntimeError(
            f"Only {len(selected)} distinct AbLang2 variants available for "
            f"{n_model_variants} shortlist slots; increase --pll-per-seed "
            "and --preselect-per-seed"
        )
    manual = reference[reference.name.isin(
        list(MEASURED_MUTATIONS) + list(RATIONAL_MUTATIONS)
    )].copy()
    manual = manual[[
        "name", "evidence_type", "mutation_count", "mutations", "sequence",
        "dataset", "study_KD_nM", "same_study_parent_KD_nM",
        "study_fold_improvement",
    ]]

    proposed = selected[[
        "seed", "new_mutation", "new_position", "mutation_count",
        "mutations", "sequence", "masked_log_odds", "confidence",
        "confidence_delta_vs_seed", "pseudo_log_likelihood",
        "seed_pseudo_log_likelihood", "pll_delta_vs_seed",
    ]].copy()
    proposed.insert(0, "name", [f"ML_{i:02d}" for i in range(1, len(proposed) + 1)])
    proposed.insert(1, "evidence_type", "AbLang2_unvalidated")
    proposed["origin_seed"] = proposed.pop("seed")
    manual["origin_seed"] = ""

    shortlist = pd.concat([manual, proposed], ignore_index=True)
    if shortlist.sequence.duplicated().any():
        raise AssertionError("Boltz-2 shortlist contains duplicate sequences")
    if len(shortlist) != args.shortlist_size:
        raise AssertionError("Unexpected Boltz-2 shortlist size")
    write_csv(shortlist, args.output_dir /
              f"04_boltz2_shortlist_{args.shortlist_size}.csv")
    print("Completed AbLang2 stage. Next: Boltz-2 + ipSAE on the shortlist, "
          "then select the final 15; WT and SPR2 are separate controls.")


if __name__ == "__main__":
    main()
