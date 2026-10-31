# HER2 scFv pipeline on Modal

Two files:
- `envs.py`: the three environments (Modal images), volumes and the remote functions. Changes rarely.
- `pipeline.py`: how the stages are chained (filters, ranking, which stages run). Edit this one to change the flow.

`her2_pipeline.py` is the earlier single-file version, kept as a backup. It uses the old command-line options and does not read the CONFIG block.

The tools, each in its own Modal image:

| Stage | Environment | What it does | GPU |
|---|---|---|---|
| 1 Propose | `abmpnn_image` (ProteinMPNN code + `abmpnn.pt` + 1N8Z) | Redesigns antibody residues near HER2 with HER2 fixed as context | L4 |
| 2 Score | `ablang_image` (`ablang2`, paired model) | Masks each mutated position and compares the new residue with the parent residue | L4 |
| 3 Fold | `boltz_image` (`boltz[cuda]` + Dunbrack ipSAE) | Cofolds scFv + HER2 (Thr23-Thr652, from UniProt P04626) and computes ipSAE | A100-80GB |

## Setup (once)
```
pip install -U modal
modal setup
```
Put the parent scFv (VH-GGGGSGGGGSGGGGS-VL) in `parent_scfv.txt`. The file contains trastuzumab
written from memory (242 aa, which matches the slide). Compare it with the sequence on the
starting-point slide before a real run.

## Run
```
modal run pipeline.py --stage check   # smoke test of all three images, about 1 fold
modal run pipeline.py                 # full pipeline
```

## Settings
All settings are in the `CONFIG` block at the top of `pipeline.py`. Edit it and run again.
For a one-off change without editing the file:
```
modal run pipeline.py --overrides "n_samples=20;n_fold=6;chunk=3"
```
Each run saves the settings it used as `config.json` in its `runs/<timestamp>/` folder.

| Group | Settings |
|---|---|
| Which residues change | `auto_design`, `contact_cutoff`, `design_vh`, `design_vl`, `fixed_vh`, `fixed_vl`, `protect_cysteines`, `omit_aas` |
| AbMPNN sampling | `n_samples`, `temps`, `seed_design` |
| Filters before folding | `min_mutations`, `max_mutations`, `min_ablang_mean`, `min_ablang_min`, `n_fold` |
| Boltz-2 and ipSAE | `chunk`, `seed_fold`, `diffusion_samples`, `recycling_steps`, `sampling_steps`, `ipsae_pae_cutoff`, `ipsae_dist_cutoff`, `use_template`, `template_force`, `template_threshold` |
| Shortlist | `min_ipsae`, `final_n`, `shortlist_weights` |

Positions in `design_*` and `fixed_*` are 1-based indices into your parent VH or VL (not IMGT or Kabat),
and ranges work: `"26-35, 50, 95-102"`. Designable residues = (contact residues if `auto_design`) + `design_*` - `fixed_*`.
Set `auto_design=False` to design only the positions you list.

Outputs go to `runs/<timestamp>/` on your machine:
`1_candidates_all.csv` (every AbMPNN sequence), `1_candidates.csv` (after the mutation filter), `2_scored.csv`,
`3_folded.csv` (includes the parent), `4_shortlist.csv` and `config.json`.
Predicted structures and confidence files are in the Modal volume `her2-results`.

## Using the 1N8Z crystal structure as a Boltz-2 template
If the parent's ipSAE comes out near 0, Boltz-2 is not finding the binding pose by itself. Try:
```
modal run pipeline.py --stage check --overrides "use_template=True"
modal run pipeline.py --stage check --overrides "use_template=True;template_force=True;template_threshold=2"
```
`use_template` gives Boltz-2 the 1N8Z structure as a template. With `template_force`, a potential also keeps the
backbone within `template_threshold` A of it. The template is off by default. The first run after enabling the
code rebuilds the Boltz image once, to add the 1N8Z file.
The parent will score high because it is the template, so judge the variants against it. If the competition's
own Boltz-2 run uses no template, your ipSAE values will not match theirs.

## How the design works
- By default, designable positions are antibody residues with a backbone atom within `contact_cutoff` of HER2
  (8 A, so roughly first and second shell). Cysteines are never changed and never proposed.
- Variants are rebuilt from your parent scFv, so only designed positions differ from it.
- Candidates are kept if they have `min_mutations` to `max_mutations` mutations (5 to 14 by default).
- Candidates below `min_ablang_mean` or `min_ablang_min` (off by default) are dropped, then the top `n_fold` by AbLang2 mean delta go to Boltz-2.
  `2_scored.csv` is saved before this filter, so you can pick cutoffs from it. The parent is folded first as the baseline.
- The scFv chain is folded without an MSA, and HER2 uses one cached MSA, so scores are comparable.
- `4_shortlist.csv` is a weighted rank sum of ipSAE and AbLang2 delta (`shortlist_weights`). The final 15 are your call.

## Editing tips
- Every image in `envs.py` ends with `.add_local_python_source("envs")`. Keep it, or the cloud containers cannot import `envs`.
- Changing `pipeline.py` never rebuilds an image. Changing an image's install lines does, once.
- To add a stage (for example ESM scoring), add an image and a function in `envs.py`, import it in `pipeline.py`, and call it in `main`.

## Things to check on the first run (tools I could not run from here)
1. `--stage check` prints the AbMPNN checkpoint keys. If `num_edges` is missing, the script fills in 48 and warns.
2. The FASTA parser matches output segments to chains by exact native sequence. It raises an error if they do not match.
3. AbLang2 masking assumes `*` is the mask token and `|` separates heavy and light. A length check catches a mismatch.
4. Boltz flags `--write_full_pae` and `msa: empty`, and whether the ipSAE script reads Boltz-2 output files.
5. Pin an exact `boltz` version in `boltz_image` once a run works.

## Cost
Set a Modal spend limit first. At list prices an A100-80GB is about $2.50 per hour, so 50 folds at 1 to 2 minutes each is a few dollars.
Run `--stage check` before the full run, and lower `--n-fold` to start small.

## Not included
ESM2/ESM3 scoring and DSM proposals. They fit as extra images with the same pattern.
