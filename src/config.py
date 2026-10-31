# Model configuration
MODEL_PATH = "Synthyra/DSM_ppi_full"

# Generation settings
TEMPERATURE = 1.0
REMASKING = "random"
STEP_DIVISOR = 10
SAFE_MODE = True

# Candidate generation
NUM_CANDIDATES = 100
RANDOM_SEED = 42

# Sequence constraints
MIN_MUTATIONS = 5
EXPECTED_BINDER_LENGTH = 242

# Output paths
OUTPUT_FASTA = "outputs/dsm_ppi_candidates.fasta"
OUTPUT_METADATA = "outputs/generation_metadata.csv"