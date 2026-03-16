from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output files
COMBINED_CORPUS_FILE = PROCESSED_DATA_DIR / "corpus.txt"

# Cleaning settings
MIN_LINE_LENGTH = 2
MAX_CONSECUTIVE_BLANKS = 1
LOWERCASE_TEXT = False
