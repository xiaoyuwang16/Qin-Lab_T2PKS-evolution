from pathlib import Path

# Window sizes for embedding
WINDOW_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200]

# Color scheme for visualization
COLOR_SCHEME = {
    'TypeII/III': '#FF6B6B',
    'FormIvariants': '#4ECDC4',  
    'Anaero': '#FFBE0B',
    'FormIAB/CD': '#9B5DE5',
    'Thermus': '#00F5D4',
    'Other': '#808080'
}

# KNN Graph parameters
KNN_K = 20
KNN_THRESHOLD = 0.5

# PCA parameters
PCA_COMPONENTS = 200 

# Base directories
BASE_DIR = Path("/YOURPATH/Qin-Lab_T2PKS-evolution/example")
OUTPUT_DIR = BASE_DIR / "output"
