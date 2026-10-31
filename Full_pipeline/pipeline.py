"""
pipeline.py: every setting and how the stages are chained
=========================================================
Change the CONFIG block below to change what the pipeline does. The environments
and remote functions live in envs.py and normally stay untouched.

    pip install -U modal && modal setup
    modal run pipeline.py --stage check                       # cheap smoke test of all 3 images
    modal run pipeline.py                                     # full run with CONFIG as written
    modal run pipeline.py --overrides "n_samples=20;n_fold=6;chunk=3"   # one-off changes

Positions are 1-based indices into YOUR parent VH or VL sequence (not IMGT or Kabat numbers).
Ranges are fine: "26-35, 50, 95-102".
"""
import ast
import csv
import datetime
import json
from pathlib import Path

from envs import (
    abmpnn_design,
    ablang_score,
    app,
    boltz_fold,
    inspect_abmpnn,
    split_scfv,
)

CONFIG = dict(
    # ---------------------------------------------------------------- inputs
    parent_file="parent_scfv.txt",   # parent scFv: VH + GGGGSGGGGSGGGGS + VL
    antigen_chain="C",               # HER2 chain letter in 1N8Z

    # ------------------------------------- stage 1: AbMPNN, which residues change
    auto_design=True,                # True: designable = residues within contact_cutoff of HER2
    contact_cutoff=8.0,              # Angstrom, backbone-atom distance to HER2
    design_vh="",                    # extra VH positions to design, e.g. "26-35, 50-65, 95-102"
    design_vl="",                    # extra VL positions to design
    fixed_vh="",                     # VH positions that must never change (wins over design_vh)
    fixed_vl="",                     # VL positions that must never change
    protect_cysteines=True,          # never change a cysteine
    omit_aas="CX",                   # amino acids AbMPNN may never propose
    n_samples=100,                   # sequences per temperature (use a multiple of 10)
    temps="0.1 0.2 0.3",             # sampling temperatures; higher = more diverse
    seed_design=37,

    # ---------------------------------------------------- filters before folding
    min_mutations=5,                 # challenge rule
    max_mutations=14,
    min_ablang_mean=None,            # keep variants with AbLang2 mean delta >= this (None = no cutoff)
    min_ablang_min=None,             # ... and worst single mutation >= this (None = no cutoff)
    n_fold=48,                       # fold at most this many, best AbLang2 mean first (None = all)

    # ---------------------------------------------------- stage 3: Boltz-2, ipSAE
    chunk=6,                         # variants per Boltz call
    seed_fold=1,
    diffusion_samples=1,             # structures per variant; the best by ipSAE is kept
    recycling_steps=3,
    sampling_steps=None,             # None = Boltz default
    ipsae_pae_cutoff=10,
    ipsae_dist_cutoff=10,
    use_template=False,              # True: give Boltz-2 the 1N8Z crystal structure as a template
    template_force=False,            # True: keep the backbone within template_threshold of the template
    template_threshold=2.0,          # Angstrom, only used when template_force is True

    # ------------------------------------------------------------- shortlist
    min_ipsae=None,                  # drop folded variants below this ipSAE (None = keep all)
    final_n=15,
    shortlist_weights={"ipsae": 1.0, "ablang_delta_mean": 1.0},   # rank-sum weights
)


# -------------------------------------------------------------- helpers
def _apply_overrides(base, overrides):
    """'a=1;b=2.5;design_vh=26-35,50' -> a copy of CONFIG with those values replaced."""
    cfg = dict(base)
    for part in [p for p in overrides.split(";") if p.strip()]:
        key, _, raw = part.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in cfg:
            raise SystemExit(f"Unknown setting '{key}'. Valid settings: {', '.join(cfg)}")
        if isinstance(cfg[key], str):
            cfg[key] = raw
        else:
            try:
                cfg[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                cfg[key] = raw
    return cfg


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


# ------------------------------------------------------------- the chain
@app.local_entrypoint()
def main(stage: str = "full", overrides: str = ""):
    cfg = _apply_overrides(CONFIG, overrides)
    parent_scfv = Path(cfg["parent_file"]).read_text().split()[0].upper()
    parent_vh, parent_vl = split_scfv(parent_scfv)
    print(f"Parent: VH {len(parent_vh)} aa, VL {len(parent_vl)} aa, scFv {len(parent_scfv)} aa")
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    abmpnn_cfg = {k: cfg[k] for k in (
        "antigen_chain", "auto_design", "contact_cutoff", "design_vh", "design_vl", "fixed_vh",
        "fixed_vl", "protect_cysteines", "omit_aas", "n_samples", "temps")}
    abmpnn_cfg["seed"] = cfg["seed_design"]
    boltz_cfg = {k: cfg[k] for k in (
        "diffusion_samples", "recycling_steps", "sampling_steps", "ipsae_pae_cutoff", "ipsae_dist_cutoff",
        "use_template", "template_force", "template_threshold")}
    boltz_cfg["seed"] = cfg["seed_fold"]

    if stage == "check":
        print("AbMPNN checkpoint:", inspect_abmpnn.remote())
        probe = {"id": "probe", "vh": parent_vh[:30] + "A" + parent_vh[31:], "vl": parent_vl}
        print("AbLang2 probe:", ablang_score.remote(parent_vh, parent_vl, [probe]))
        print("Boltz-2 parent fold:",
              boltz_fold.remote([{"id": "parent", "scfv": parent_scfv}], run_id, boltz_cfg))
        print("All three environments ran. Now run without --stage check.")
        return

    # 1) propose
    designed = abmpnn_design.remote(parent_vh, parent_vl, abmpnn_cfg)
    cands = designed["variants"]
    print(f"AbMPNN: {len(cands)} unique sequences "
          f"({designed['n_design_vh']} VH and {designed['n_design_vl']} VL positions designable)")
    _write_csv(out_dir / "1_candidates_all.csv", cands)
    cands = [c for c in cands if cfg["min_mutations"] <= c["n_mut"] <= cfg["max_mutations"]]
    print(f"{len(cands)} candidates with {cfg['min_mutations']}-{cfg['max_mutations']} mutations")
    if not cands:
        raise SystemExit("No candidates passed the mutation filter. Widen the designable positions "
                         "(contact_cutoff, design_vh/vl) or raise temps.")
    _write_csv(out_dir / "1_candidates.csv", cands)

    # 2) score, then filter. The unfiltered scores are saved first so you can pick cutoffs from them.
    scores = {s["id"]: s for s in ablang_score.remote(parent_vh, parent_vl, cands)}
    for c in cands:
        c.update(scores[c["id"]])
    _write_csv(out_dir / "2_scored.csv", cands)
    if cfg["min_ablang_mean"] is not None:
        cands = [c for c in cands if c["ablang_delta_mean"] >= cfg["min_ablang_mean"]]
    if cfg["min_ablang_min"] is not None:
        cands = [c for c in cands if c["ablang_delta_min"] >= cfg["min_ablang_min"]]
    print(f"{len(cands)} candidates pass the AbLang2 cutoffs")
    if not cands:
        raise SystemExit("No candidates passed the AbLang2 cutoffs. See 2_scored.csv and loosen them.")
    cands.sort(key=lambda c: c["ablang_delta_mean"], reverse=True)
    to_fold = cands if cfg["n_fold"] is None else cands[:int(cfg["n_fold"])]
    print(f"Folding {len(to_fold)} variants")

    # 3) fold. The parent runs first on its own: it is the baseline and it builds the HER2 MSA cache.
    base = boltz_fold.remote([{"id": "parent", "scfv": parent_scfv}], run_id, boltz_cfg)[0]
    print(f"Parent baseline: ipSAE {base['ipsae']}, ipTM {base['iptm']}")
    size = int(cfg["chunk"])
    batches = [[{"id": c["id"], "scfv": c["scfv"]} for c in to_fold[i:i + size]]
               for i in range(0, len(to_fold), size)]
    folded = {}
    for res in boltz_fold.starmap([(b, run_id, boltz_cfg) for b in batches]):
        for r in res:
            folded[r["id"]] = r
    rows = [{**c, **{k: v for k, v in folded[c["id"]].items() if k != "id"}} for c in to_fold]
    parent_row = {"id": "parent", "scfv": parent_scfv, "n_mut": 0, **{k: v for k, v in base.items() if k != "id"}}
    _write_csv(out_dir / "3_folded.csv", rows + [parent_row])

    # 4) shortlist: weighted rank sum (lower is better). The final pick is your call.
    var = [r for r in rows if cfg["min_ipsae"] is None or (r.get("ipsae") or -1) >= cfg["min_ipsae"]]
    if not var:
        raise SystemExit("No folded variant reached min_ipsae. See 3_folded.csv.")
    ranks = {k: _rank(var, k, True) for k in cfg["shortlist_weights"]}
    order = sorted(range(len(var)), key=lambda i: sum(w * ranks[k][i] for k, w in cfg["shortlist_weights"].items()))
    shortlist = [var[i] for i in order[:int(cfg["final_n"])]]
    _write_csv(out_dir / "4_shortlist.csv", [parent_row] + shortlist)
    print(f"Done. Results in {out_dir}/ (structures on the Modal volume 'her2-results' under {run_id}/)")
    print(f"Parent ipSAE {base['ipsae']}; shortlist best {shortlist[0]['ipsae']}")
