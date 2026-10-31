"""
HER2 scFv design pipeline on Modal
==================================
AbMPNN (propose)  ->  AbLang2 (score)  ->  Boltz-2 + ipSAE (fold and judge)

Each model lives in its own Modal image (own Python environment), so their
dependencies never clash. The local entrypoint at the bottom chains them.

    pip install -U modal && modal setup
    modal run her2_pipeline.py --stage check     # cheap smoke test of all 3 images
    modal run her2_pipeline.py                   # full pipeline

Everything marked CHECK depends on a tool detail I could not test from here.
The `--stage check` run is there to surface those before you spend credits.
"""
import csv
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import urllib.request
import uuid
from pathlib import Path

import modal

app = modal.App("her2-scfv-pipeline")

# ---------------------------------------------------------------- constants
LINKER = "GGGGSGGGGSGGGGS"
HER2_UNIPROT = "P04626"
HER2_START, HER2_END = 23, 652      # 1-based inclusive, from the challenge slide
FOLD_GPU = "A100-80GB"              # cheaper option to try: "L40S"
MAX_FOLD_CONTAINERS = 4             # parallel Boltz containers

# ------------------------------------------------------------------ volumes
# boltz weights + the HER2 MSA are cached here so they download once.
boltz_cache = modal.Volume.from_name("her2-boltz-cache", create_if_missing=True)
# predicted structures and confidence files of every folded variant.
results_vol = modal.Volume.from_name("her2-results", create_if_missing=True)

# ------------------------------------------------- environment 1: AbMPNN
abmpnn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("torch", "numpy", "biopython")
    .run_commands(
        "git clone --depth 1 https://github.com/dauparas/ProteinMPNN /opt/ProteinMPNN",
        "mkdir -p /opt/abmpnn /opt/structures",
        # AbMPNN weights, Zenodo record 8164693 (CC BY 4.0)
        "curl -fL -o /opt/abmpnn/abmpnn.pt 'https://zenodo.org/records/8164693/files/abmpnn.pt?download=1'",
        "curl -fL -o /opt/structures/1N8Z.pdb https://files.rcsb.org/download/1N8Z.pdb",
        "ls -l /opt/abmpnn /opt/structures",
    )
)

# ------------------------------------------------- environment 2: AbLang2
ablang_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "ablang2")
    # download the paired model weights once, at build time
    .run_commands(
        "python -c \"import ablang2; ablang2.pretrained(model_to_use='ablang2-paired', "
        "random_init=False, ncpu=1, device='cpu')\""
    )
)

# ---------------------------------------- environment 3: Boltz-2 and ipSAE
boltz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("boltz[cuda]")  # pin an exact version here once a run works
    .run_commands("git clone --depth 1 https://github.com/DunbrackLab/IPSAE /opt/IPSAE")
)


# ------------------------------------------------------- shared pure helpers
def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def split_scfv(scfv: str):
    parts = scfv.strip().upper().split(LINKER)
    if len(parts) != 2:
        raise ValueError(
            f"Expected exactly one {LINKER} linker in the scFv, found {len(parts) - 1}."
        )
    return parts[0], parts[1]


def list_mutations(parent_vh, parent_vl, vh, vl):
    """Mutation labels such as 'VH:T28A'. Positions are 1-based in the Fv chain."""
    out = []
    for chain, p, v in (("VH", parent_vh, vh), ("VL", parent_vl, vl)):
        if len(p) != len(v):
            raise ValueError(f"{chain} length changed ({len(p)} -> {len(v)}); only substitutions are supported.")
        out += [f"{chain}:{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(p, v)) if a != b]
    return out


# ============================================================ STAGE 1: AbMPNN
def _align_map(pdb_seq, parent_seq):
    """Map 0-based index in the PDB chain -> 0-based index in the parent Fv (local alignment)."""
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -6
    aligner.extend_gap_score = -1
    aln = aligner.align(pdb_seq, parent_seq)[0]
    mapping = {}
    for (t0, t1), (q0, q1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(int(t1 - t0)):
            mapping[int(t0) + k] = int(q0) + k
    return mapping, float(aln.score)


def _contact_positions(rec, ab_chain, antigen_chain, cutoff):
    """0-based residues of ab_chain whose backbone atoms lie within `cutoff` A of the antigen."""
    import numpy as np

    def backbone(ch):
        c = rec[f"coords_chain_{ch}"]
        return np.stack(
            [np.array(c[f"{a}_chain_{ch}"], dtype=float) for a in ("N", "CA", "C", "O")], axis=1
        )

    ab = backbone(ab_chain)  # (L, 4, 3)
    ag = backbone(antigen_chain).reshape(-1, 3)
    ag = ag[~np.isnan(ag).any(axis=1)]
    d = np.linalg.norm(ab[:, :, None, :] - ag[None, None, :, :], axis=-1)
    dmin = np.nanmin(d.reshape(d.shape[0], -1), axis=1)
    return {int(i) for i in np.where(dmin <= cutoff)[0]}


def _prepare_abmpnn_weights():
    """ProteinMPNN's run script expects model_state_dict + num_edges in the checkpoint."""
    import torch

    ckpt = torch.load("/opt/abmpnn/abmpnn.pt", map_location="cpu", weights_only=False)
    patched = False
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
        ckpt = {"model_state_dict": ckpt}
        patched = True
    if "num_edges" not in ckpt:
        ckpt["num_edges"] = 48  # ProteinMPNN default
        patched = True
    if "noise_level" not in ckpt:
        ckpt["noise_level"] = 0.0
        patched = True
    if patched:
        print("WARNING: abmpnn.pt lacked ProteinMPNN checkpoint fields; filled in defaults "
              "(num_edges=48). Check the AbMPNN paper if results look off.")
    os.makedirs("/opt/abmpnn_run", exist_ok=True)
    torch.save(ckpt, "/opt/abmpnn_run/abmpnn.pt")


@app.function(image=abmpnn_image)
def inspect_abmpnn() -> dict:
    """Smoke test: what does the AbMPNN checkpoint contain?"""
    import torch

    ckpt = torch.load("/opt/abmpnn/abmpnn.pt", map_location="cpu", weights_only=False)
    info = {"type": type(ckpt).__name__}
    if isinstance(ckpt, dict):
        info["keys"] = list(ckpt.keys())[:12]
        for k in ("num_edges", "noise_level"):
            if k in ckpt:
                info[k] = ckpt[k]
    print(info)
    return info


@app.function(image=abmpnn_image, gpu="L4", timeout=1800)
def abmpnn_design(
    parent_vh: str,
    parent_vl: str,
    antigen_chain: str = "C",
    contact_cutoff: float = 8.0,
    n_samples: int = 100,
    temps: str = "0.1 0.2 0.3",
    seed: int = 37,
) -> dict:
    """Redesign antibody residues near HER2 on the 1N8Z backbone, HER2 kept fixed as context.

    Returns scFv variants where ONLY the designed positions differ from the parent.
    """
    work = Path("/tmp/abmpnn")
    shutil.rmtree(work, ignore_errors=True)
    (work / "pdb").mkdir(parents=True)
    shutil.copy("/opt/structures/1N8Z.pdb", work / "pdb" / "1N8Z.pdb")
    _prepare_abmpnn_weights()

    parsed = work / "parsed.jsonl"
    run(["python", "/opt/ProteinMPNN/helper_scripts/parse_multiple_chains.py",
         "--input_path", work / "pdb", "--output_path", parsed])
    rec = json.loads(parsed.read_text().splitlines()[0])
    name = rec["name"]
    chains = sorted(k[len("seq_chain_"):] for k in rec if k.startswith("seq_chain_"))
    ab_chains = [c for c in chains if c != antigen_chain]
    if len(ab_chains) != 2 or antigen_chain not in chains:
        raise RuntimeError(f"Unexpected chains in the structure: {chains}")

    # which antibody chain is heavy and which is light, decided by sequence not by letter
    scores = {c: (_align_map(rec[f"seq_chain_{c}"], parent_vh)[1],
                  _align_map(rec[f"seq_chain_{c}"], parent_vl)[1]) for c in ab_chains}
    vh_chain = max(ab_chains, key=lambda c: scores[c][0] - scores[c][1])
    vl_chain = [c for c in ab_chains if c != vh_chain][0]
    parent_of = {vh_chain: parent_vh, vl_chain: parent_vl}

    mapping, design = {}, {}
    for ch in (vh_chain, vl_chain):
        seq = rec[f"seq_chain_{ch}"]
        mapping[ch], _ = _align_map(seq, parent_of[ch])
        ident = sum(seq[i] == parent_of[ch][j] for i, j in mapping[ch].items()) / max(len(mapping[ch]), 1)
        if ident < 0.9:
            print(f"WARNING: chain {ch} is only {ident:.0%} identical to the parent over the aligned region.")
        contacts = _contact_positions(rec, ch, antigen_chain, contact_cutoff)
        # keep: inside the variable domain, and never touch cysteines
        design[ch] = sorted(i for i in contacts if i in mapping[ch] and parent_of[ch][mapping[ch][i]] != "C")
    print(f"Heavy chain {vh_chain}: {len(design[vh_chain])} designable positions; "
          f"light chain {vl_chain}: {len(design[vl_chain])}")

    # ProteinMPNN control files. Fixed positions are 1-based indices into the chain sequence.
    chain_id = {name: [[vh_chain, vl_chain], [antigen_chain]]}
    fixed = {name: {ch: [i + 1 for i in range(len(rec[f"seq_chain_{ch}"])) if i not in set(design[ch])]
                    for ch in (vh_chain, vl_chain)}}
    (work / "chain_id.jsonl").write_text(json.dumps(chain_id) + "\n")
    (work / "fixed.jsonl").write_text(json.dumps(fixed) + "\n")

    batch = 10 if n_samples % 10 == 0 else 1
    run(["python", "/opt/ProteinMPNN/protein_mpnn_run.py",
         "--jsonl_path", parsed,
         "--chain_id_jsonl", work / "chain_id.jsonl",
         "--fixed_positions_jsonl", work / "fixed.jsonl",
         "--out_folder", work / "out",
         "--num_seq_per_target", n_samples,
         "--sampling_temp", temps,
         "--batch_size", batch,
         "--seed", seed,
         "--omit_AAs", "CX",
         "--path_to_model_weights", "/opt/abmpnn_run",
         "--model_name", "abmpnn"])

    # parse the FASTA: record 0 is the native sequence, the rest are designs.
    # CHECK: multi-chain records are '/'-separated; segments are matched to chains by exact native sequence.
    recs, header = [], None
    for line in (work / "out" / "seqs" / f"{name}.fa").read_text().splitlines():
        if line.startswith(">"):
            header = line[1:]
        elif header is not None:
            recs.append((header, line.strip()))
            header = None
    native_segments = recs[0][1].split("/")
    seg_chain = []
    for seg in native_segments:
        match = [c for c in (vh_chain, vl_chain) if rec[f"seq_chain_{c}"] == seg]
        if not match:
            raise RuntimeError(
                "Could not match FASTA segments to chains "
                f"(segment lengths {[len(s) for s in native_segments]}). Open the FASTA and adjust the parser."
            )
        seg_chain.append(match[0])

    variants, seen = [], set()
    for header, seq in recs[1:]:
        segs = dict(zip(seg_chain, seq.split("/")))
        new = {}
        for ch in (vh_chain, vl_chain):
            fv = list(parent_of[ch])
            for i in design[ch]:
                fv[mapping[ch][i]] = segs[ch][i]
            new[ch] = "".join(fv)
        vh, vl = new[vh_chain], new[vl_chain]
        if (vh, vl) in seen:
            continue
        seen.add((vh, vl))
        mut = list_mutations(parent_vh, parent_vl, vh, vl)
        t = re.search(r"\bT=([0-9.]+)", header)
        s = re.search(r"(?<![A-Za-z_])score=([0-9.]+)", header)
        variants.append({
            "id": f"ab{len(variants):04d}",
            "vh": vh, "vl": vl, "scfv": vh + LINKER + vl,
            "n_mut": len(mut), "mutations": " ".join(mut),
            "temperature": float(t.group(1)) if t else None,
            "mpnn_score": float(s.group(1)) if s else None,
        })
    return {
        "variants": variants,
        "heavy_chain": vh_chain, "light_chain": vl_chain,
        "n_design_vh": len(design[vh_chain]), "n_design_vl": len(design[vl_chain]),
    }


# =========================================================== STAGE 2: AbLang2
@app.function(image=ablang_image, gpu="L4", timeout=1800)
def ablang_score(parent_vh: str, parent_vl: str, variants: list) -> list:
    """Score each variant with AbLang2 (paired model).

    For every mutated position: mask it, read the model's log-probability of the new residue
    minus that of the parent residue. Positive = the model likes the swap.
    CHECK: relies on ablang.tokenizer(...) and ablang.AbLang(tokens) as shown in the AbLang2 README,
    and on '*' being the mask token and '|' the heavy/light separator.
    """
    import torch
    import ablang2

    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device="cuda")
    try:
        ablang.AbLang.eval()
    except Exception:
        pass

    def tokens(strs):
        return ablang.tokenizer(strs, pad=True, w_extra_tkns=False, device="cuda")

    # vocabulary: token id of each amino acid, read off the tokenizer itself
    aa_id = {a: int(tokens([f"{a}|{a}"])[0, 0]) for a in "ACDEFGHIKLMNPQRSTVWY"}
    parent = parent_vh + "|" + parent_vl

    out = []
    for v in variants:
        s = v["vh"] + "|" + v["vl"]
        pos = [i for i, (a, b) in enumerate(zip(s, parent)) if a != b]
        row = {"id": v["id"]}
        if not pos:
            row.update(ablang_delta_sum=0.0, ablang_delta_mean=0.0, ablang_delta_min=0.0)
            out.append(row)
            continue
        masked = [s[:i] + "*" + s[i + 1:] for i in pos]
        t = tokens(masked)
        if t.shape[1] != len(s):
            raise RuntimeError(f"Token length {t.shape[1]} != sequence length {len(s)}; "
                               "check the tokenizer settings (w_extra_tkns).")
        with torch.no_grad():
            out_t = ablang.AbLang(t).float()
        if out_t.shape[1] != len(s):
            raise RuntimeError(f"Model output length {out_t.shape[1]} != {len(s)}.")
        probs_like = bool((out_t.min() >= 0) and ((out_t.sum(-1) - 1).abs().max() < 1e-3))
        logp = out_t.clamp_min(1e-9).log() if probs_like else torch.log_softmax(out_t, dim=-1)
        deltas = [float(logp[k, i, aa_id[s[i]]] - logp[k, i, aa_id[parent[i]]]) for k, i in enumerate(pos)]
        row.update(ablang_delta_sum=sum(deltas), ablang_delta_mean=sum(deltas) / len(deltas),
                   ablang_delta_min=min(deltas))
        out.append(row)
    return out


# ===================================================== STAGE 3: Boltz-2 + ipSAE
MSA_DIR = Path("/cache/msa")


def _get_her2() -> str:
    cache = Path("/cache/her2_ecd.txt")
    if cache.exists():
        return cache.read_text().strip()
    url = f"https://rest.uniprot.org/uniprotkb/{HER2_UNIPROT}.fasta"
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode()
    full = "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))
    ecd = full[HER2_START - 1:HER2_END]
    if not (len(ecd) == HER2_END - HER2_START + 1 and ecd[0] == "T" and ecd[-1] == "T"):
        raise RuntimeError("HER2 slice does not look like Thr23..Thr652; check the UniProt entry.")
    cache.write_text(ecd)
    return ecd


def _cached_her2_msa():
    for ext in ("a3m", "csv"):
        p = MSA_DIR / f"her2.{ext}"
        if p.exists():
            return str(p)
    return None


def _harvest_msa(out_dir):
    cands = [p for p in glob.glob(f"{out_dir}/**/msa/*", recursive=True) if p.endswith((".a3m", ".csv"))]
    if cands:
        MSA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(cands[0], MSA_DIR / f"her2.{cands[0].rsplit('.', 1)[1]}")


def _boltz_yaml(scfv, her2, her2_msa):
    # CHECK: `msa: empty` runs a chain without an MSA (documented for Boltz). The scFv gets no MSA
    # and HER2 gets one MSA shared by every variant, so scores stay comparable.
    lines = ["version: 1", "sequences:",
             "  - protein:", "      id: A", f"      sequence: {scfv}", "      msa: empty",
             "  - protein:", "      id: B", f"      sequence: {her2}"]
    if her2_msa:
        lines.append(f"      msa: {her2_msa}")
    return "\n".join(lines) + "\n"


def _parse_ipsae(txt_path):
    header, rows = None, []
    for line in Path(txt_path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Chn1":
            header = parts
        elif header and len(parts) >= len(header) - 1:
            rows.append(dict(zip(header, parts)))
    pair = [r for r in rows if {r.get("Chn1"), r.get("Chn2")} == {"A", "B"}]
    best = [r for r in pair if r.get("Type") == "max"] or pair
    return max(float(r["ipSAE"]) for r in best) if best else None


@app.function(
    image=boltz_image, gpu=FOLD_GPU, timeout=3600,
    volumes={"/cache": boltz_cache, "/results": results_vol},
    max_containers=MAX_FOLD_CONTAINERS,
)
def boltz_fold(items: list, run_id: str, seed: int = 1, diffusion_samples: int = 1) -> list:
    """Cofold each scFv with the HER2 ectodomain and compute ipSAE.

    items: [{"id": ..., "scfv": ...}, ...]; the whole list goes through one `boltz predict` call.
    """
    boltz_cache.reload()
    her2 = _get_her2()
    msa = _cached_her2_msa()

    work = Path(f"/tmp/boltz_{uuid.uuid4().hex[:8]}")
    (work / "in").mkdir(parents=True)
    for it in items:
        (work / "in" / f"{it['id']}.yaml").write_text(_boltz_yaml(it["scfv"], her2, msa))

    cmd = ["boltz", "predict", work / "in", "--out_dir", work / "out", "--cache", "/cache/boltz",
           "--diffusion_samples", diffusion_samples, "--seed", seed,
           "--write_full_pae", "--output_format", "mmcif"]
    if msa is None:
        cmd.append("--use_msa_server")  # first call only: builds the HER2 MSA, which is then cached
    run(cmd)

    if msa is None:
        _harvest_msa(work / "out")
    boltz_cache.commit()

    ipsae_py = glob.glob("/opt/IPSAE/**/ipsae.py", recursive=True)[0]
    results = []
    for it in items:
        best = None
        for cif in sorted(glob.glob(f"{work}/out/**/{it['id']}_model_*.cif", recursive=True)):
            k = re.search(r"_model_(\d+)\.cif$", cif).group(1)
            d = os.path.dirname(cif)
            pae, conf = f"{d}/pae_{it['id']}_model_{k}.npz", f"{d}/confidence_{it['id']}_model_{k}.json"
            run(["python", ipsae_py, pae, cif, 10, 10])
            ipsae = _parse_ipsae(f"{cif[:-4]}_10_10.txt")
            c = json.loads(Path(conf).read_text())
            row = {"id": it["id"], "model": int(k), "ipsae": ipsae,
                   "iptm": c.get("iptm"), "ptm": c.get("ptm"),
                   "complex_plddt": c.get("complex_plddt"), "complex_iplddt": c.get("complex_iplddt"),
                   "confidence_score": c.get("confidence_score")}
            if best is None or (ipsae or -1) > (best["ipsae"] or -1):
                best = row
                keep = Path(f"/results/{run_id}/{it['id']}")
                keep.mkdir(parents=True, exist_ok=True)
                for f in (cif, pae, conf, f"{cif[:-4]}_10_10.txt"):
                    shutil.copy(f, keep)
        if best is None:
            raise RuntimeError(f"No Boltz prediction found for {it['id']} under {work}/out")
        results.append(best)
    results_vol.commit()
    return results


# ============================================================ local entrypoint
def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _rank(rows, key, reverse):
    def val(i):
        v = rows[i].get(key)
        return float("-inf") if v is None else v

    order = sorted(range(len(rows)), key=val, reverse=reverse)
    return {i: r for r, i in enumerate(order)}


@app.local_entrypoint()
def main(
    stage: str = "full",                 # "check" = cheap smoke test, "full" = whole pipeline
    parent_file: str = "parent_scfv.txt",
    antigen_chain: str = "C",
    n_samples: int = 100,                # AbMPNN samples per temperature
    temps: str = "0.1 0.2 0.3",
    contact_cutoff: float = 8.0,         # Angstrom, backbone-atom distance to HER2
    min_mutations: int = 5,              # challenge rule
    max_mutations: int = 14,
    n_fold: int = 48,                    # how many variants go to Boltz-2
    final_n: int = 15,
    chunk: int = 6,                      # variants per Boltz call
    seed: int = 1,
):
    parent_scfv = Path(parent_file).read_text().split()[0].upper()
    parent_vh, parent_vl = split_scfv(parent_scfv)
    print(f"Parent: VH {len(parent_vh)} aa, VL {len(parent_vl)} aa, scFv {len(parent_scfv)} aa")
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if stage == "check":
        print("AbMPNN checkpoint:", inspect_abmpnn.remote())
        probe = {"id": "probe", "vh": parent_vh[:30] + "A" + parent_vh[31:], "vl": parent_vl}
        print("AbLang2 probe:", ablang_score.remote(parent_vh, parent_vl, [probe]))
        print("Boltz-2 parent fold:", boltz_fold.remote([{"id": "parent", "scfv": parent_scfv}], run_id, seed))
        print("All three environments ran. Now run without --stage check.")
        return

    # 1) propose
    designed = abmpnn_design.remote(parent_vh, parent_vl, antigen_chain, contact_cutoff, n_samples, temps)
    cands = designed["variants"]
    print(f"AbMPNN: {len(cands)} unique sequences "
          f"({designed['n_design_vh']} VH and {designed['n_design_vl']} VL positions designable)")
    cands = [c for c in cands if min_mutations <= c["n_mut"] <= max_mutations]
    print(f"{len(cands)} candidates with {min_mutations}-{max_mutations} mutations")
    if not cands:
        raise SystemExit("No candidates passed the mutation filter. Raise --contact-cutoff or --temps.")
    _write_csv(out_dir / "1_candidates.csv", cands)

    # 2) score
    scores = {s["id"]: s for s in ablang_score.remote(parent_vh, parent_vl, cands)}
    for c in cands:
        c.update(scores[c["id"]])
    _write_csv(out_dir / "2_scored.csv", cands)
    cands.sort(key=lambda c: c["ablang_delta_mean"], reverse=True)
    to_fold = cands[:n_fold]
    print(f"Folding the top {len(to_fold)} by AbLang2 mean delta")

    # 3) fold. The parent runs first on its own: it is the baseline and it builds the HER2 MSA cache.
    base = boltz_fold.remote([{"id": "parent", "scfv": parent_scfv}], run_id, seed)[0]
    print(f"Parent baseline: ipSAE {base['ipsae']}, ipTM {base['iptm']}")
    batches = [[{"id": c["id"], "scfv": c["scfv"]} for c in to_fold[i:i + chunk]]
               for i in range(0, len(to_fold), chunk)]
    folded = {}
    for res in boltz_fold.starmap([(b, run_id, seed) for b in batches]):
        for r in res:
            folded[r["id"]] = r
    rows = [{**c, **{k: v for k, v in folded[c["id"]].items() if k != "id"}} for c in to_fold]
    rows.append({"id": "parent", "scfv": parent_scfv, "n_mut": 0, **{k: v for k, v in base.items() if k != "id"}})
    _write_csv(out_dir / "3_folded.csv", rows)

    # 4) shortlist: simple rank sum of ipSAE (high) and AbLang2 mean delta (high). Your call from here.
    var = [r for r in rows if r["id"] != "parent"]
    r1, r2 = _rank(var, "ipsae", True), _rank(var, "ablang_delta_mean", True)
    order = sorted(range(len(var)), key=lambda i: r1[i] + r2[i])
    shortlist = [var[i] for i in order[:final_n]]
    _write_csv(out_dir / "4_shortlist.csv", [rows[-1]] + shortlist)
    print(f"Done. Results in {out_dir}/ (structures on the Modal volume 'her2-results' under {run_id}/)")
    print(f"Parent ipSAE {base['ipsae']}; shortlist best {shortlist[0]['ipsae']}")
