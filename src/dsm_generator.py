
"""Generate HER2-conditioned scFv variants using three masking strategies."""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DSM_PATH = PROJECT_ROOT / "external" / "DSM"
sys.path.insert(0, str(DSM_PATH))

from models.modeling_dsm import DSM

from config import (
    MODEL_PATH,
    TEMPERATURE,
    REMASKING,
    STEP_DIVISOR,
    SAFE_MODE,
    NUM_CANDIDATES,
    RANDOM_SEED,
    MIN_MUTATIONS,
)

from sequence_utils import mask_positions, count_mutations
from mutation_strategies import (
    STRATEGIES,
    get_mask_positions,
    validate_parent_layout,
)

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    """Read a single-sequence FASTA file."""
    lines = Path(path).read_text().splitlines()
    sequence = "".join(
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(">")
    ).upper()

    if not sequence:
        raise ValueError(f"No sequence found in {path}")

    return sequence


def is_valid_candidate(parent, candidate, masked_positions):
    """Check sequence validity and ensure only masked positions changed."""
    if len(candidate) != len(parent):
        return False, "wrong_length"

    if not candidate or not set(candidate).issubset(AMINO_ACIDS):
        return False, "invalid_amino_acids_or_leftover_mask"

    mutations = count_mutations(parent, candidate)
    if mutations < MIN_MUTATIONS:
        return False, "fewer_than_minimum_mutations"

    allowed = set(masked_positions)
    for index, (original, new) in enumerate(zip(parent, candidate)):
        if index not in allowed and original != new:
            return False, "unmasked_position_changed"

    return True, "valid"


class DSMGenerator:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading {model_path} on {self.device}")

        self.model = DSM.from_pretrained(model_path).to(self.device).eval()
        self.tokenizer = self.model.tokenizer

    def generate(
        self,
        target_sequence,
        parent_sequence,
        positions_to_mask,
        num_candidates=NUM_CANDIDATES,
    ):
        """Generate variants for one masking strategy."""

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        masked_binder, _ = mask_positions(
            parent_sequence, positions_to_mask
        )

        eos = self.tokenizer.eos_token
        template = target_sequence + eos + masked_binder

        template_tokens = self.tokenizer.encode(
            template,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        print(f"Encoded input length: {template_tokens.shape[-1]} tokens")

        generated = []

        for index in range(num_candidates):
            input_tokens = template_tokens.clone()

            with torch.no_grad():
                output_tokens = self.model.mask_diffusion_generate(
                    tokenizer=self.tokenizer,
                    input_tokens=input_tokens,
                    step_divisor=STEP_DIVISOR,
                    temperature=TEMPERATURE,
                    remasking=REMASKING,
                    preview=False,
                    slow=False,
                    safe_mode=SAFE_MODE,
                )

            # Retain the decoding call used in the DSM PPI demo.
            # We still need to verify its second argument's expected meaning
            # before treating these results as final.
            attention_mask = torch.ones_like(output_tokens)

            target_out, binder_out = self.model.decode_dual_input(
                output_tokens,
                attention_mask,
                eos,
            )
            generated.append({
                "candidate": binder_out[0],
                "target": target_out[0],
                "generation_index": index,
            })

            print(f"Generated {index + 1}/{num_candidates}")

        return generated


def save_candidates(parent_sequence, generated, strategy_name, masked_positions):
    """Validate and save one strategy's candidates."""

    valid_sequences = []
    seen = set()
    rejection_counts = {}

    for item in generated:
        candidate = item["candidate"]
        is_valid, reason = is_valid_candidate(
            parent_sequence, candidate, masked_positions
        )

        if not is_valid:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        if candidate in seen:
            rejection_counts["duplicate"] = (
                rejection_counts.get("duplicate", 0) + 1
            )
            continue

        seen.add(candidate)
        valid_sequences.append(candidate)

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output_dir / f"{strategy_name}_candidates.fasta"
    csv_path = output_dir / f"{strategy_name}_metadata.csv"

    with open(fasta_path, "w") as fasta:
        for index, sequence in enumerate(valid_sequences, start=1):
            fasta.write(f">{strategy_name}_candidate_{index}\n{sequence}\n")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["candidate_id", "sequence", "mutation_count"],
        )
        writer.writeheader()

        for index, sequence in enumerate(valid_sequences, start=1):
            writer.writerow({
                "candidate_id": f"{strategy_name}_candidate_{index}",
                "sequence": sequence,
                "mutation_count": count_mutations(
                    parent_sequence, sequence
                ),
            })

    print(f"\nResults for {strategy_name}")
    print(f"Generated: {len(generated)}")
    print(f"Valid unique candidates: {len(valid_sequences)}")
    print(f"Rejected: {rejection_counts}")
    print(f"FASTA: {fasta_path}")
    print(f"Metadata: {csv_path}")

    return valid_sequences


def main():
    target = read_fasta(PROJECT_ROOT / "data" / "HER2_ECD.fasta")
    parent = read_fasta(PROJECT_ROOT / "data" / "parent_scFv.fasta")

    validate_parent_layout(parent)

    if len(target) != 630:
        raise ValueError(f"Expected HER2 ECD length 630, got {len(target)}")

    print(f"HER2 ECD length: {len(target)}")
    print(f"Parent scFv length: {len(parent)}")

    generator = DSMGenerator()

    for strategy in STRATEGIES:
        name = strategy["name"]
        positions = get_mask_positions(strategy)

        print("\n" + "=" * 60)
        print(f"Running strategy: {name}")
        print(f"Masked residues: {len(positions)}")
        print("=" * 60)

        generated = generator.generate(
            target_sequence=target,
            parent_sequence=parent,
            positions_to_mask=positions,
            num_candidates=NUM_CANDIDATES,
        )

        save_candidates(parent, generated, name, positions)


if __name__ == "__main__":
    main()
