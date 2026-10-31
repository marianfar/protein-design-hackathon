#!/usr/bin/env python3
"""Launch the HER2–scFv Boltz-2 eval.py pipeline on a Modal H100 GPU.

Files required in the same project directory:
  eval.py
  ablang2_results/00_reference_sequences.csv
  ablang2_results/04_boltz2_shortlist_30.csv (required for variants)

Setup on your Mac (inside the trastuzumab conda environment):
  python -m pip install modal
  modal setup

Usage:
  modal run modal_eval.py --wt-only      # WT baseline first
  modal run modal_eval.py --limit 1      # smoke test: only first variant
  modal run modal_eval.py --limit 30     # full 30-variant batch
  modal run modal_eval.py --limit 30 --msa-mode server
  modal run modal_eval.py --wt-only --score-only   # REUSE existing WT; NO GPU
  modal run modal_eval.py --wt-only --no-kernels   # GPU fallback if CUDA kernel fails
  modal run modal_eval.py --limit 30 --score-only --download-variants  # fetch all variant files; NO GPU

Ranked results are downloaded to ./boltz_results/<msa_mode>/<wt|variants>/
Variant structures and matrices remain in the persistent Modal Volume unless
--download-variants is passed; WT raw files are downloaded by default.
The volume is
`trastuzumab-boltz-results` under <msa_mode>/<wt|variants>/.
One H100 is requested per invocation, with a 1-hour function timeout.
This is NOT a dollar spending cap: set a workspace budget in Modal billing.
"""

from pathlib import Path
import subprocess

import modal

# Modal imports this module twice: locally from the user's project folder, and
# remotely as /root/modal_eval.py. The attached files live in /root/project.
# Use the project directory that actually contains eval.py in each environment.
CLOUD_PROJECT = Path("/root/project")
BASE = CLOUD_PROJECT if (CLOUD_PROJECT / "eval.py").is_file() else Path(__file__).resolve().parent
SHORTLIST = BASE / "ablang2_results" / "04_boltz2_shortlist_30.csv"
REFERENCE = BASE / "ablang2_results" / "00_reference_sequences.csv"
if not REFERENCE.is_file():
    raise FileNotFoundError(f"Missing AbLang2 reference sequences: {REFERENCE}")

app = modal.App("trastuzumab-her2-boltz2")
results_volume = modal.Volume.from_name("trastuzumab-boltz-results", create_if_missing=True)
weights_volume = modal.Volume.from_name("trastuzumab-boltz-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boltz[cuda]==2.2.1", "numpy")
    .add_local_file(str(BASE / "eval.py"), "/root/project/eval.py")
    .add_local_file(str(REFERENCE), "/root/project/ablang2_results/00_reference_sequences.csv")
)
if SHORTLIST.is_file():
    image = image.add_local_file(str(SHORTLIST), "/root/project/ablang2_results/04_boltz2_shortlist_30.csv")


# Scoring existing predictions requires only NumPy, not Boltz or a GPU.
score_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4")
    .add_local_file(str(BASE / "eval.py"), "/root/project/eval.py")
    .add_local_file(str(REFERENCE), "/root/project/ablang2_results/00_reference_sequences.csv")
)
if SHORTLIST.is_file():
    score_image = score_image.add_local_file(
        str(SHORTLIST), "/root/project/ablang2_results/04_boltz2_shortlist_30.csv"
    )


def wt_raw_paths(msa_mode: str) -> list[str]:
    """Paths relative to the mounted Modal results volume, for WT download."""
    output = Path("/root/project/boltz_results")
    directory = output / msa_mode / "wt"
    files = []
    for folder in [directory / "predictions" / "WT", *directory.glob("boltz_results_*/predictions/WT")]:
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if (f.name.startswith(("pae_WT_model_", "confidence_WT_model_", "WT_model_"))
                    and f.suffix in {".npz", ".json", ".cif"}):
                files.append(str(f.relative_to(output)))
    return files


def variant_raw_paths(msa_mode: str) -> list[str]:
    """Locate all existing variant PAE, confidence and structure files in the Modal Volume."""
    output = Path("/root/project/boltz_results")
    directory = output / msa_mode / "variants"
    prediction_roots = [directory / "predictions", *sorted(directory.glob("boltz_results_*/predictions"))]
    files = []
    for prediction_root in prediction_roots:
        if not prediction_root.is_dir():
            continue
        for variant_dir in sorted(prediction_root.iterdir()):
            if not variant_dir.is_dir():
                continue
            for f in sorted(variant_dir.iterdir()):
                if f.is_file() and f.suffix in {".npz", ".json", ".cif"} and (
                    f.name.startswith(("pae_", "confidence_")) or f.suffix == ".cif"
                ):
                    files.append(str(f.relative_to(output)))
    if not files:
        raise FileNotFoundError(f"No variant prediction files found under {directory}")
    return sorted(set(files))


@app.function(
    image=score_image,
    timeout=300,
    volumes={"/root/project/boltz_results": results_volume},
)
def score_existing(limit: int, msa_mode: str, wt_only: bool, download_variants: bool) -> list[str]:
    """Score completed Boltz predictions on CPU WITHOUT rerunning Boltz."""
    kind = "wt" if wt_only else "variants"
    command = [
        "python", "eval.py", "--score-only", "--limit", str(limit),
        "--out-dir", f"boltz_results/{msa_mode}/{kind}",
    ]
    if wt_only:
        command.append("--wt-only")
    try:
        subprocess.run(command, cwd="/root/project", check=True)
    finally:
        results_volume.commit()
    return wt_raw_paths(msa_mode) if wt_only else (variant_raw_paths(msa_mode) if download_variants else [])



@app.function(
    image=image,
    gpu="H100",
    timeout=3600,  # One hour per invocation; does not cap overall dollar spend.
    volumes={
        "/root/project/boltz_results": results_volume,
        "/root/.boltz": weights_volume,
    },
)
def evaluate(limit: int, msa_mode: str, override: bool, wt_only: bool, no_kernels: bool, download_variants: bool):
    command = [
        "python", "eval.py", "--run", "--limit", str(limit),
        "--msa-mode", msa_mode,
        "--out-dir", f"boltz_results/{msa_mode}/{'wt' if wt_only else 'variants'}",
    ]
    if wt_only:
        command.append("--wt-only")
    if override:
        command.append("--override")
    if no_kernels:
        command.append("--no-kernels")
    try:
        subprocess.run(command, cwd="/root/project", check=True)
    finally:
        # Persist partial predictions so reruns can reuse completed jobs.
        results_volume.commit()
        weights_volume.commit()
    return wt_raw_paths(msa_mode) if wt_only else (variant_raw_paths(msa_mode) if download_variants else [])


@app.local_entrypoint()
def main(limit: int = 1, msa_mode: str = "empty", override: bool = False, wt_only: bool = False, no_kernels: bool = False, score_only: bool = False, download_variants: bool = False):
    if not wt_only and not SHORTLIST.is_file():
        raise FileNotFoundError(f"Missing AbLang2 shortlist: {SHORTLIST}")
    if not (1 <= limit <= 30):
        raise ValueError("--limit must be from 1 to 30")
    if msa_mode not in ("empty", "server"):
        raise ValueError("--msa-mode must be 'empty' or 'server'")
    if score_only and override:
        raise ValueError("--override is incompatible with --score-only")
    if download_variants and wt_only:
        raise ValueError("--download-variants is for variants; omit --wt-only")
    if score_only:
        print("Scoring existing Modal Volume results on CPU; no GPU requested.")
        raw_files = score_existing.remote(limit, msa_mode, wt_only, download_variants)
    else:
        raw_files = evaluate.remote(limit, msa_mode, override, wt_only, no_kernels, download_variants)
    run_kind = "wt" if wt_only else "variants"
    local_output = BASE / "boltz_results" / msa_mode / run_kind
    local_output.mkdir(parents=True, exist_ok=True)
    paths = [f"{msa_mode}/{run_kind}/{filename}" for filename in
             ("ranked_ipsae.csv", "all_model_ipsae.csv")]
    paths.extend(raw_files)
    for rel in paths:
        dest = BASE / "boltz_results" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = b"".join(results_volume.read_file(rel))
        dest.write_bytes(content)
        print(f"Downloaded {dest}")
    print(f"Completed. Downloaded {len(raw_files)} raw prediction files. "
          "Use --download-variants --score-only to download all variant structures without GPU.")
