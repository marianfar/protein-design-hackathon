"""
envs.py: the three Modal environments and the remote functions that run in them
===============================================================================
AbMPNN (propose)  ->  AbLang2 (score)  ->  Boltz-2 + ipSAE (fold and judge)

Each model lives in its own Modal image (own Python environment), so their
dependencies never clash. This file changes rarely. How the stages are chained
lives in pipeline.py, which imports `app` and the functions from here.

Every image ends with .add_local_python_source("envs") so the cloud containers
can import this module. Keep that line when you edit an image.

Everything marked CHECK depends on a tool detail I could not test from here.
`modal run pipeline.py --stage check` is there to surface those before you spend credits.
"""
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
    .add_local_python_source("envs")
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
    .add_local_python_source("envs")
)

# ---------------------------------------- environment 3: Boltz-2 and ipSAE
boltz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("boltz[cuda]")  # pin an exact version here once a run works
    .run_commands("git clone --depth 1 https://github.com/DunbrackLab/IPSAE /opt/IPSAE")
    .run_commands(
        "mkdir -p /opt/structures",
        "python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://files.rcsb.org/download/1N8Z.cif', '/opt/structures/1N8Z.cif')\"",
        "ls -l /opt/structures",
    )
    .add_local_python_source("envs")
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


def parse_positions(spec, length):
    """'26-35, 50, 95-102' (or a list of ints) -> sorted 1-based positions within 1..length."""
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple, set)):
        items = [str(x) for x in spec]
    else:
        items = [p for p in re.split(r"[,\s]+", str(spec).strip()) if p]
    out = set()
    for it in items:
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", it)
        if not m:
            raise ValueError(f"Cannot read position '{it}'. Use numbers and ranges such as '26-35, 50'.")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        a, b = min(a, b), max(a, b)
        if a < 1 or b > length:
            raise ValueError(f"Position {it} is outside 1..{length}.")
        out.update(range(a, b + 1))
    return sorted(out)


def select_design_positions(auto_pdb_idx, mapping, parent_seq, design_spec, fixed_spec, protect_cysteines):
    """Decide which residues AbMPNN may change.

    auto_pdb_idx: 0-based residues found by the contact search (indices into the PDB chain)
    mapping:      PDB chain index -> parent Fv index (both 0-based)
    design_spec / fixed_spec: extra positions to design / never touch, 1-based in the PARENT Fv
    Fixed always wins over design. Returns (PDB chain indices, parent positions with no coordinates).
    """
    plen = len(parent_seq)
    inv = {j: i for i, j in mapping.items()}
    chosen = {mapping[i] + 1 for i in auto_pdb_idx if i in mapping}
    chosen |= set(parse_positions(design_spec, plen))
    chosen -= set(parse_positions(fixed_spec, plen))
    if protect_cysteines:
        chosen = {p for p in chosen if parent_seq[p - 1] != "C"}
    skipped = sorted(p for p in chosen if (p - 1) not in inv)
    return sorted(inv[p - 1] for p in chosen if (p - 1) in inv), skipped


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


ABMPNN_DEFAULTS = dict(
    antigen_chain="C",
    auto_design=True,        # designable = residues within contact_cutoff of the antigen
    contact_cutoff=8.0,      # Angstrom, backbone-atom distance
    design_vh="", design_vl="",   # extra positions to design, 1-based in the parent VH / VL
    fixed_vh="", fixed_vl="",     # positions that must never change (these win over design_*)
    protect_cysteines=True,
    omit_aas="CX",           # amino acids AbMPNN may never propose
    n_samples=100,           # sequences per temperature
    temps="0.1 0.2 0.3",
    seed=37,
)


@app.function(image=abmpnn_image, gpu="L4", timeout=1800)
def abmpnn_design(parent_vh: str, parent_vl: str, cfg: dict = None) -> dict:
    """Redesign antibody residues on the 1N8Z backbone, HER2 kept fixed as context.

    cfg overrides ABMPNN_DEFAULTS. Returns scFv variants where ONLY the designed positions
    differ from the parent.
    """
    cfg = {**ABMPNN_DEFAULTS, **(cfg or {})}
    antigen_chain, n_samples, temps = cfg["antigen_chain"], int(cfg["n_samples"]), str(cfg["temps"])
    seed, omit_aas = int(cfg["seed"]), str(cfg["omit_aas"])
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
        label = "vh" if ch == vh_chain else "vl"
        auto = _contact_positions(rec, ch, antigen_chain, float(cfg["contact_cutoff"])) if cfg["auto_design"] else set()
        design[ch], skipped = select_design_positions(
            auto, mapping[ch], parent_of[ch], cfg[f"design_{label}"], cfg[f"fixed_{label}"], cfg["protect_cysteines"])
        if skipped:
            print(f"WARNING: {label.upper()} positions {skipped} have no coordinates in 1N8Z and were skipped.")
    shown = {("VH" if ch == vh_chain else "VL"): sorted(mapping[ch][i] + 1 for i in design[ch])
             for ch in (vh_chain, vl_chain)}
    print(f"Designable positions (1-based in the parent Fv): {shown}")

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
         "--omit_AAs", omit_aas,
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
        "design_positions": shown,
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
TEMPLATE_CIF = "/opt/structures/1N8Z.cif"


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


def _boltz_yaml(scfv, her2, her2_msa, template=None):
    # CHECK: `msa: empty` runs a chain without an MSA (documented for Boltz). The scFv gets no MSA
    # and HER2 gets one MSA shared by every variant, so scores stay comparable.
    lines = ["version: 1", "sequences:",
             "  - protein:", "      id: A", f"      sequence: {scfv}", "      msa: empty",
             "  - protein:", "      id: B", f"      sequence: {her2}"]
    if her2_msa:
        lines.append(f"      msa: {her2_msa}")
    if template is not None:
        # CHECK: with only a cif path, Boltz picks the best-matching template chain for each query chain.
        # The scFv is one query chain, so it will be matched to the heavy or the light template chain.
        lines += ["templates:", f"  - cif: {TEMPLATE_CIF}"]
        if template["force"]:
            lines += ["    force: true", f"    threshold: {template['threshold']}"]
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


BOLTZ_DEFAULTS = dict(
    seed=1,
    diffusion_samples=1,     # structures per input; the best by ipSAE is kept
    recycling_steps=3,       # Boltz default
    sampling_steps=None,     # None = Boltz default
    ipsae_pae_cutoff=10,
    ipsae_dist_cutoff=10,
    use_template=False,      # give Boltz-2 the 1N8Z crystal structure as a template
    template_force=False,    # True: keep the backbone within template_threshold of the template
    template_threshold=2.0,  # Angstrom, only used when template_force is True
)


def _num(x):
    return int(x) if float(x) == int(float(x)) else float(x)


@app.function(
    image=boltz_image, gpu=FOLD_GPU, timeout=3600,
    volumes={"/cache": boltz_cache, "/results": results_vol},
    max_containers=MAX_FOLD_CONTAINERS,
)
def boltz_fold(items: list, run_id: str, cfg: dict = None) -> list:
    """Cofold each scFv with the HER2 ectodomain and compute ipSAE.

    items: [{"id": ..., "scfv": ...}, ...]; the whole list goes through one `boltz predict` call.
    """
    cfg = {**BOLTZ_DEFAULTS, **(cfg or {})}
    pae_cut, dist_cut = _num(cfg["ipsae_pae_cutoff"]), _num(cfg["ipsae_dist_cutoff"])
    boltz_cache.reload()
    her2 = _get_her2()
    msa = _cached_her2_msa()

    work = Path(f"/tmp/boltz_{uuid.uuid4().hex[:8]}")
    (work / "in").mkdir(parents=True)
    template = ({"force": bool(cfg["template_force"]), "threshold": float(cfg["template_threshold"])}
                if cfg["use_template"] else None)
    for it in items:
        (work / "in" / f"{it['id']}.yaml").write_text(_boltz_yaml(it["scfv"], her2, msa, template))

    cmd = ["boltz", "predict", work / "in", "--out_dir", work / "out", "--cache", "/cache/boltz",
           "--diffusion_samples", int(cfg["diffusion_samples"]), "--seed", int(cfg["seed"]),
           "--recycling_steps", int(cfg["recycling_steps"]),
           "--write_full_pae", "--output_format", "mmcif"]
    if cfg["sampling_steps"] is not None:
        cmd += ["--sampling_steps", int(cfg["sampling_steps"])]
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
            run(["python", ipsae_py, pae, cif, pae_cut, dist_cut])
            txts = [p for p in glob.glob(f"{cif[:-4]}_*.txt") if "byres" not in p]
            if not txts:
                raise RuntimeError(f"ipSAE wrote no summary file next to {cif}")
            ipsae_txt = txts[0]
            ipsae = _parse_ipsae(ipsae_txt)
            c = json.loads(Path(conf).read_text())
            row = {"id": it["id"], "model": int(k), "ipsae": ipsae,
                   "iptm": c.get("iptm"), "ptm": c.get("ptm"),
                   "complex_plddt": c.get("complex_plddt"), "complex_iplddt": c.get("complex_iplddt"),
                   "confidence_score": c.get("confidence_score")}
            if best is None or (ipsae or -1) > (best["ipsae"] or -1):
                best = row
                keep = Path(f"/results/{run_id}/{it['id']}")
                keep.mkdir(parents=True, exist_ok=True)
                for f in (cif, pae, conf, ipsae_txt):
                    shutil.copy(f, keep)
        if best is None:
            raise RuntimeError(f"No Boltz prediction found for {it['id']} under {work}/out")
        results.append(best)
    results_vol.commit()
    return results
