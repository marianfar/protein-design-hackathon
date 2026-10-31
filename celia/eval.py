#!/usr/bin/env python3
"""Run Boltz-2 HER2/trastuzumab-scFv cofolding and calculate ipSAEmin.

Input: ablang2_results/04_boltz2_shortlist_30.csv, with name and sequence columns.
Target: human HER2 (ERBB2, UniProt P04626) extracellular residues 23-652.

This script needs numpy for scoring, and the `boltz` command on a GPU machine
for prediction. It does NOT require the original EGFR JSON/PDB files.

Examples (run from Trastuzumab_Hackathon/):
    python eval.py --prepare-only
    python eval.py --run --limit 1           # GPU smoke test
    python eval.py --run                     # process all 30; cached jobs reused
    python eval.py --score-only              # score predictions already on disk
    python eval.py --run --msa-mode server   # alternative: HER2 MSA server

Reported ipSAE is the Dunbrack d0res formulation: max per-residue mean of
1/(1+(PAE/d0(n))**2) over opposing-chain residues with PAE < cutoff.
The primary ipSAEmin column is min(A->B, B->A). For two canonical protein
chains, Boltz-2's PAE tokens are in YAML order, one token per residue.

Reference: https://github.com/DunbrackLab/IPSAE (Dunbrack Lab, MIT license).
Unlike the original IPSAE script, this script computes ipSAE only, not pDockQ,
LIS or distance-based interface summaries. It validates PAE token dimensions.

WARNING: ipSAE is a model confidence metric, not a KD or experimental affinity.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

# Human HER2/ERBB2 UniProt P04626, full-length sequence positions 1-660;
# the executable below extracts the extracellular domain (23-652).
# https://rest.uniprot.org/uniprotkb/P04626.fasta
HER2_N_TERM = (
    "MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNL"
    "ELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNG"
    "DPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLA"
    "LTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQC"
    "AAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACP"
    "YNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSAN"
    "IQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITGYLYISAWPDSLP"
    "DLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTV"
    "PWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQFLRGQEC"
    "VEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARC"
    "PSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVG"
)
HER2_ECD = HER2_N_TERM[22:652]
AA = set("ACDEFGHIKLMNPQRSTVWY")
VALID_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def read_variants(csv_path: Path, include_wt: bool, wt_only: bool = False) -> list[dict[str, str]]:
    if not csv_path.is_file() and not wt_only:
        raise FileNotFoundError(
            f"Can't find AbLang2 shortlist: {csv_path}\n"
            "Run trastuzumab_ablang2_design.py without --dry-run first."
        )
    variants = []
    if not wt_only:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"name", "sequence"}.issubset(reader.fieldnames or []):
                raise ValueError("Shortlist needs 'name' and 'sequence' CSV columns")
            variants = list(reader)

    if include_wt or wt_only:
        reference = csv_path.parent / "00_reference_sequences.csv"
        if not reference.is_file():
            raise FileNotFoundError(f"Cannot include WT: missing {reference}")
        with reference.open(newline="") as handle:
            wt = next((row for row in csv.DictReader(handle)
                       if row.get("name") == "WT"), None)
        if wt is None:
            raise ValueError("WT not present in reference CSV")
        variants.append(wt)

    if not variants:
        raise ValueError("CSV is empty")
    ids = set()
    seqs = set()
    for row in variants:
        name = (row.get("name") or "").strip()
        seq = (row.get("sequence") or "").strip().upper()
        if not VALID_ID.fullmatch(name):
            raise ValueError(f"Invalid variant name {name!r} (use letters, digits, _ or -)")
        if name in ids:
            raise ValueError(f"Duplicate variant name: {name}")
        if not (len(seq) == 242 and set(seq) <= AA):
            raise ValueError(f"{name}: expected the 242-aa canonical scFv sequence; got {len(seq)}")
        if seq in seqs:
            raise ValueError(f"Duplicate scFv sequence: {name}")
        if seq[120:135] != "GGGGS" * 3:
            raise ValueError(f"{name}: linker has changed")
        row["name"], row["sequence"] = name, seq
        ids.add(name)
        seqs.add(seq)
    return variants


def write_yaml(path: Path, scfv: str, her2: str, msa_mode: str) -> None:
    # The antibody is artificial (VH-linker-VL). We deliberately do not send
    # it to the MSA server, even if HER2 uses an MSA.
    her2_msa_line = "      msa: empty\n" if msa_mode == "empty" else ""
    content = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {scfv}\n"
        "      msa: empty\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {her2}\n"
        f"{her2_msa_line}"
    )
    path.write_text(content)


def d0_for_n(n: np.ndarray) -> np.ndarray:
    """Dunbrack ipSAE d0res for protein interfaces; 1.0 A minimum."""
    n = np.maximum(np.asarray(n, dtype=np.float64), 26.0)
    return np.maximum(1.0, 1.24 * np.cbrt(n - 15.0) - 1.8)


def directed_ipsae(pae: np.ndarray, pae_cutoff: float) -> dict[str, float | int]:
    """Rows are residues of chain1; columns are residues of chain2."""
    mask = pae < pae_cutoff
    n_good = mask.sum(axis=1)
    d0 = d0_for_n(n_good)
    contributions = (1.0 / (1.0 + (pae / d0[:, None]) ** 2)) * mask
    row_scores = np.divide(
        contributions.sum(axis=1), n_good,
        out=np.zeros(len(n_good), dtype=np.float64),
        where=n_good > 0,
    )
    best = int(np.argmax(row_scores))
    return {
        "ipsae": float(row_scores[best]),
        "n_good_for_best": int(n_good[best]),
        "best_residue_1based": best + 1,
        "n_interface_residues_from": int((n_good > 0).sum()),
        "n_good_pairs": int(n_good.sum()),
    }


def calculate_ipsae(pae_path: Path, n_antibody: int, n_her2: int,
                    pae_cutoff: float) -> dict[str, float | int]:
    with np.load(pae_path, allow_pickle=False) as f:
        if "pae" not in f:
            raise ValueError(f"{pae_path.name}: NPZ does not contain the 'pae' array")
        pae = np.asarray(f["pae"], dtype=np.float64)
    expected = n_antibody + n_her2
    if pae.shape != (expected, expected):
        raise ValueError(
            f"{pae_path.name}: PAE shape {pae.shape}, expected ({expected}, {expected}). "
            "Check target length, chain order and whether Boltz has extra tokens."
        )
    if not np.isfinite(pae).all():
        raise ValueError(f"{pae_path.name}: PAE has NaN/Inf")
    ab_to_her2 = directed_ipsae(pae[:n_antibody, n_antibody:], pae_cutoff)
    her2_to_ab = directed_ipsae(pae[n_antibody:, :n_antibody], pae_cutoff)
    a, b = ab_to_her2["ipsae"], her2_to_ab["ipsae"]
    return {
        "ipsae_ab_to_her2": a,
        "ipsae_her2_to_ab": b,
        "ipsae_min": min(a, b),
        "ipsae_max": max(a, b),
        "best_ab_residue": ab_to_her2["best_residue_1based"],
        "best_her2_uniprot_position": 22 + her2_to_ab["best_residue_1based"],
        "good_pae_pairs_ab_to_her2": ab_to_her2["n_good_pairs"],
        "good_pae_pairs_her2_to_ab": her2_to_ab["n_good_pairs"],
        "ab_residues_with_good_pae": ab_to_her2["n_interface_residues_from"],
        "her2_residues_with_good_pae": her2_to_ab["n_interface_residues_from"],
    }


def find_model_files(out_dir: Path, name: str) -> list[tuple[int, Path, Path, Path]]:
    """Locate Boltz outputs, including its boltz_results_<input> subfolder.

    Boltz-2 writes under out_dir/boltz_results_<input_folder>/predictions/name,
    NOT necessarily directly under out_dir/predictions/name.
    """
    folders = [out_dir / "predictions" / name]
    folders += sorted(out_dir.glob(f"boltz_results_*/predictions/{name}"))
    found = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for pae in folder.glob(f"pae_{name}_model_*.npz"):
            match = re.fullmatch(re.escape(f"pae_{name}_model_") + r"(\d+)\.npz", pae.name)
            if match is None:
                continue
            model = int(match.group(1))
            conf = folder / f"confidence_{name}_model_{model}.json"
            cif = folder / f"{name}_model_{model}.cif"
            found.append((model, pae, conf, cif))
    return sorted(found, key=lambda item: (item[0], str(item[1])))


def score_all(variants: list[dict[str, str]], out_dir: Path,
              pae_cutoff: float, n_her2: int) -> None:
    summary_rows = []
    all_model_rows = []
    missing = 0
    for variant in variants:
        name = variant["name"]
        models = find_model_files(out_dir, name)
        if not models:
            summary_rows.append({**variant, "status": "not_predicted"})
            missing += 1
            continue
        successful = []
        errors = []
        for model_idx, pae, confidence, cif in models:
            try:
                metrics = calculate_ipsae(pae, len(variant["sequence"]), n_her2,
                                          pae_cutoff)
                if not confidence.is_file():
                    raise FileNotFoundError(f"Missing confidence JSON: {confidence}")
                if not cif.is_file():
                    raise FileNotFoundError(f"Missing predicted structure: {cif}")
                conf = json.loads(confidence.read_text())
                model_row = {
                    **variant, **metrics,
                    "model_index": model_idx,
                    "iptm": conf.get("iptm", ""),
                    "protein_iptm": conf.get("protein_iptm", ""),
                    "complex_plddt": conf.get("complex_plddt", ""),
                    "confidence_score": conf.get("confidence_score", ""),
                    "pae_path": str(pae),
                    "structure_path": str(cif),
                    "status": "ok",
                    "pae_cutoff_angstrom": pae_cutoff,
                }
                successful.append(model_row)
                all_model_rows.append(model_row)
            except Exception as exc:
                errors.append(f"model {model_idx}: {exc}")
        if successful:
            # Model 0 is Boltz's highest-confidence sample. Keep it as the
            # primary score; using the max across samples would bias ranking.
            selected = next((r for r in successful if r["model_index"] == 0),
                            successful[0])
            summary_rows.append(selected)
            print(f"  {name:8}  ipSAEmin={selected['ipsae_min']:.4f} "
                  f"ipTM={selected['iptm']}")
        else:
            summary_rows.append({**variant, "status": "scoring_error", "error": "; ".join(errors)})
            print(f"  {name:8}  scoring error: {'; '.join(errors)}", file=sys.stderr)
    summary_rows.sort(key=lambda r: (
        r.get("status") == "ok",
        float(r.get("ipsae_min", -1)),
    ), reverse=True)
    for rank, r in enumerate(summary_rows, 1):
        r["rank_ipsae_min"] = rank if r.get("status") == "ok" else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "ranked_ipsae.csv", summary_rows)
    write_csv(out_dir / "all_model_ipsae.csv", all_model_rows)
    print(f"\nSaved {out_dir / 'ranked_ipsae.csv'}")
    print(f"Scored {len(variants) - missing} of {len(variants)} variants "
          f"(including any scoring errors).")
    if missing:
        print(f"{missing} variants have no Boltz predictions yet.")
        if missing == len(variants):
            print("Expected Boltz output under:", out_dir / "boltz_results_<input>/predictions/<variant>")
            print("Directories present:", [str(p.relative_to(out_dir)) for p in out_dir.glob("boltz_results_*")])
    if not all_model_rows:
        raise RuntimeError("No ipSAE models were scored. Check the diagnostic output above; "
                           "the scoring step cannot produce a result CSV yet.")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=PROJECT_DIR / "ablang2_results" / "04_boltz2_shortlist_30.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "boltz_results")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true",
                        help="Generate Boltz YAMLs only; safe to run locally")
    action.add_argument("--run", action="store_true",
                        help="Generate inputs, run Boltz GPU prediction, then score")
    action.add_argument("--score-only", action="store_true",
                        help="Recalculate scores from existing Boltz results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process first N variants; use --limit 1 to test")
    parser.add_argument("--include-wt", action="store_true",
                        help="Add WT from reference CSV as a control")
    parser.add_argument("--wt-only", action="store_true",
                        help="Run only WT from reference CSV; shortlist not required")
    parser.add_argument("--msa-mode", choices=["empty", "server"], default="empty",
                        help="empty: no MSA (fast); server: fetch HER2 MSA")
    parser.add_argument("--pae-cutoff", type=float, default=15.0,
                        help="PAE threshold in Angstrom; 15 used by Adaptyv")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--no-kernels", action="store_true",
                        help="Disable optional cuEquivariance kernels if CUDA imports fail")
    parser.add_argument("--override", action="store_true",
                        help="Force rerunning predictions despite existing cache")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not (0 < args.pae_cutoff <= 31):
        parser.error("--pae-cutoff must be between 0 and 31 Angstrom")
    if args.diffusion_samples < 1 or args.recycling_steps < 1 or args.sampling_steps < 1:
        parser.error("steps and diffusion-samples must be positive")
    if len(HER2_ECD) != 630 or HER2_ECD[0] != "T" or HER2_ECD[-1] != "T":
        raise AssertionError("HER2 ECD sequence extraction failed")

    variants = read_variants(args.input, args.include_wt, args.wt_only)
    if args.limit:
        variants = variants[:args.limit]
    out_dir = args.out_dir.resolve()
    print(f"Loaded {len(variants)} scFv variants, HER2 residues 23-652 "
          f"({len(HER2_ECD)} aa).")

    if args.score_only:
        score_all(variants, out_dir, args.pae_cutoff, len(HER2_ECD))
        return

    yaml_dir = out_dir / "input_yamls" / f"batch_{len(variants)}_{args.msa_mode}"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    # Remove only YAML files created in this dedicated batch folder so a
    # previously larger batch cannot be submitted inadvertently.
    for old_yaml in yaml_dir.glob("*.yaml"):
        old_yaml.unlink()
    for row in variants:
        write_yaml(yaml_dir / f"{row['name']}.yaml", row["sequence"],
                   HER2_ECD, args.msa_mode)
    print(f"Generated {len(variants)} YAML files in {yaml_dir}")
    if args.prepare_only:
        print("Preparation complete. Run with --run on a CUDA GPU machine.")
        return

    if shutil.which("boltz") is None:
        raise RuntimeError("Can't find 'boltz' on PATH. Install Boltz-2 on the GPU "
                           "machine (e.g. python -m pip install boltz).")
    cmd = [
        "boltz", "predict", str(yaml_dir),
        "--out_dir", str(out_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--recycling_steps", str(args.recycling_steps),
        "--sampling_steps", str(args.sampling_steps),
        "--diffusion_samples", str(args.diffusion_samples),
        "--write_full_pae",
    ]
    if args.msa_mode == "server":
        cmd.append("--use_msa_server")
    if args.no_kernels:
        cmd.append("--no_kernels")
    if args.override:
        cmd.append("--override")
    print("Executing:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        print(f"Boltz exited with code {result.returncode}; scoring any completed "
              "predictions before exiting.", file=sys.stderr)
    score_all(variants, out_dir, args.pae_cutoff, len(HER2_ECD))
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
