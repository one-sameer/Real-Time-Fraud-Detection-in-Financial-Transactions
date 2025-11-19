# Central configuration for directories and constants

import os

# -------------------------
# Project Root
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# Data directory
# -------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_FILE = os.path.join(DATA_DIR, "creditcard.csv")

# -------------------------
# Model directory (trained .pkl files)
# -------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# -------------------------
# Tuning hyperparameter results
# -------------------------
TUNING_DIR = os.path.join(PROJECT_ROOT, "tuning_results")

# -------------------------
# Ensure directories exist
# -------------------------
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TUNING_DIR, exist_ok=True)

# -------------------------
# Common constants
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------------
# Debug / Check paths
# -------------------------
if __name__ == "__main__":
    print("Project Root:", PROJECT_ROOT)
    print("Data Dir:", DATA_DIR)
    print("Models Dir:", MODELS_DIR)
    print("Tuning Dir:", TUNING_DIR)
    print("Data File:", DATA_FILE)