# The Evolutionary Processes of Bacterial Aromatic Polyketide Ketosynthases

This repository contains the code and data analysis scripts associated with the manuscript: "The evolutionary processes of bacterial aromatic polyketide ketosynthases".

## Abstract
The biosynthesis of bacterial aromatic polyketide polyketides (type II polyketides, T2PKs) employs a single set of catalysts (ketosynthases, KSs or KSα, with chain length factors, CLFs or KSβ) and iteratively assembles a carbon backbone with precise chain length control.

In this study, we employed MAAPE (Modular Assembly Analysis of Protein Embeddings), a recently developed algorithm based on large protein language models (PLMs), to glean insights into the evolution process of KSs and CLFs. Our findings indicated the evolutionary history of KS and CLF domains from bacterial T2PKSs and identified a shared ancestral cluster (Cluster A), supporting a common origin. Despite structural homology, KSs and CLFs followed distinct evolutionary paths, shaped by coevolution and early horizontal gene transfer (HGT).

## Features
- PLM Embeddings: Utilization of ESM-2 (Evolutionary Scale Modeling) to extract high-dimensional protein features.\
- MAAPE Algorithm: Application of the Modular Assembly Analysis of Protein Embeddings to reconstruct evolutionary trajectories.\
- Functional Prediction: Integration of DeepFRI (Deep Functional Residue Identification) to correlate evolutionary clusters with GO molecular functions.\
- Visualization: Scripts for generating phylogenetic trees, similarity heatmaps, and co-evolutionary network graphs.

## Introduction of MAAPE
Implements a five-step pipeline for analyzing protein sequence evolution relationships and constructing their similarity networks:

1. Embedding Generation\
Processes input protein sequences\
Utilizes the ESM-2 language model to generate sequence embeddings\
Normalization and dimension reduction

2. Path Generation\
Sliding window segments the embeddings into multiple smaller sub-vectors\
Identifies assembly Paths between these different window size sub-vectors\
Maps potential connections between sequence segments

3. Weight and Edge Calculation\
Computes co-assembly relationships between input sequences based on identified Paths\
Determines edge directions between sequence pairs\
Calculates edge weights based on sequence relationships\
Generates a weighted, directed edge list

4. Builds a K-nearest neighbor (KNN) similarity network\

5. Visualization\
Maps previously calculated directions and weights onto KNN edges\
Generates the final MAAPE (Molecular Assembly And Protein Engineering) network\


## MAAPE workflow diagram
![graphical](https://github.com/user-attachments/assets/77610421-6d2d-44fb-bcb0-4944b8586c5a)

## MAAPE algorithm
![MAAPE算法示意图](https://github.com/user-attachments/assets/b36e147d-d28e-4784-9292-de9e3ae33e7a)

##  Requirements
torch,
transformers,
biopython,
numpy,
tqdm,
scikit-learn,
faiss-cpu,
networkx,
matplotlib,
typing-extensions

## Installation
```bash
git clone https://github.com/xiaoyuwang16/MAAPE.git
cd MAAPE
pip install -r requirements.txt
```
## Contents and introduction of files

`/path/to/Qin-Lab_T2PKS-evolution/example` Contains the script & KS data used in our analysis:\
...\
***
├── example/\
│   ├── _1_generate_embeddings.py       # Data processing pipeline code (6 files prefixed with _number_)\
│   ├── _2_generate_paths.py\
│   └── ...\
│   └── KS_all&outgroup.fasta           # the KS sequence, both KSα and KSβ\
│   └── converted_thresholds_pca.npy    # A file containing sub-vector similarity thresholds corresponding to each window size. The methodology for calculating these thresholds is detailed in the MAAPE article.\
│   └── order_index.txt                 # Index and protein classification (outgroup/KSα/KSβ) corresponding to `KS_all&outgroup.fasta`, used for color coding during visualization.\
│   └── sequence_index.txt              # Index and protein names (ie. ksa_+ABXA-BE-24566B_13) corresponding to `KS_all&outgroup.fasta`.\
│   └── Visual_Fig2a.py                 # visualizing scripts for each figure in our paper.\
│   └── ...\
...\





 











## Data Format Requirements

Input files:\
1. Protein sequences in FASTA format, there's a `/path/to/MAAPE/example/test.fasta` in example folder which contains 110 Rubisco protein sequences\
2. Order index file: `/path/to/MAAPE/example/order_index.txt`
   Contains sequence indices and their corresponding protein categories. This information is used for node coloring in the visualization.
3. Similarity threshold file for determining whether sub-vectors of different window sizes are equivalent：`/path/to/MAAPE/example/converted_thresholds_pca.npy`，this file is derived from threshold_window_size_5 = 0.00001, with thresholds for other window sizes converted proportionally using square root scaling.
   
Output:\
Embedding files (.npy), there is a embedding file already L2 normalized and reduced to 110 dimensions: `/path/to/MAAPE/example/output/normalized_pca_embeddings.npy`\
Path information (.pkl)\
Edge weights and graph structure (.pkl, .txt)\
Visualization plots

## Configuration
Settings can be changed at `/path/to/MAAPE/constant.py`\
These are parameters:

1. WINDOW_SIZES\
Used for sliding window analysis for sequence embedding, set multiple window sizes to capture sequence features at different scales.

2. COLOR_SCHEME\
Defines distinct color codes for different protein types\
Used for visualization

3. KNN Graph Parameters\
KNN_K = 20: Sets the number of nearest neighbors for each node\
KNN_THRESHOLD = 0.5: Defines the edge weight threshold

5. PCA Parameters\
PCA_COMPONENTS = 110: Sets the number of dimensions for dimensionality reduction, can't be too small for retaining key feature information

6. Directory Settings\
BASE_DIR: Sets the base working directory\
OUTPUT_DIR: Sets the output directory



## Usage
```python
import os
import sys
maape_path = '/path/to/MAAPE' 
sys.path.append(maape_path)

import importlib.util

def import_file(file_path):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 1. Generate Embeddings
script = import_file('/path/to/MAAPE/example/_1_generate_embeddings.py')
generate_embeddings = script.main
generate_embeddings()

# 2. Generate Paths
script = import_file('/path/to/MAAPE/example/_2_generate_paths.py')
generate_paths = script.main
generate_paths()

# 3. Calculate Weights and Edges
script = import_file('/path/to/MAAPE/example/_3_calculate_weights_and_edges.py')
calculate_weights = script.main
calculate_weights()

# 4. Build and Analyze Graph
script = import_file('/path/to/MAAPE/example/_4_build_and_analyze_graph.py')
build_and_analyze = script.main
build_and_analyze()

# 5. Visualize Results
script = import_file('/path/to/MAAPE/example/_5_visualize_maape.py')
maape_visual = script.main
maape_visual()

# 6. Visualize Aggregated Results
script = import_file('/path/to/MAAPE/example/_6_aggregated_visualization.py')
aggregated_maape = script.main
aggregated_maape()
```
Step 5 & 6 will generate MAAPE graph and its condensed version.

MAAPE graph of ‘/path/to/MAAPE/example/test.fasta’:
![下载](https://github.com/user-attachments/assets/5e1489d7-51e0-4432-8167-75ebf98544d8)
Condensed graph:
![下载 (1)](https://github.com/user-attachments/assets/dcc2c80d-96a2-4f7e-b503-9e086225395f)
