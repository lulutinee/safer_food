'''
Docstring for interface
Fait l'interface entre le frontend streamlit et le backend de ml_logic
Commence par charger les paramètres du modèle d'Arrhénius dans une variable arrhenius_parameters
'''


# interface/__init__.py

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import ast


# Load environment variables
load_dotenv()

# Get path from environment
secondary_model_path = os.getenv("SECONDARY_MODEL_PATH")

if secondary_model_path is None:
    raise ValueError("SECONDARY_MODEL_PATH is not defined in .env")

# Resolve path relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
csv_path = PROJECT_ROOT / secondary_model_path

if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

# Load once
cols_to_convert = [
    "Initial Value",
    "Lag",
    "Maximum Rate",
    "Final Value"
]

arrhenius_parameters = pd.read_csv(
    csv_path, sep='\t',
    converters={col: ast.literal_eval for col in cols_to_convert}
)
