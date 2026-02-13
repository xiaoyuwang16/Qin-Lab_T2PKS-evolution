# The Evolutionary Processes of Bacterial Aromatic Polyketide Ketosynthases

This repository contains the code and data analysis scripts associated with the manuscript: "The evolutionary processes of bacterial aromatic polyketide ketosynthases".

## Abstract
The biosynthesis of bacterial aromatic polyketide polyketides (type II polyketides, T2PKs) employs a single set of catalysts (ketosynthases, KSs or KSα, with chain length factors, CLFs or KSβ) and iteratively assembles a carbon backbone with precise chain length control.

In this study, we employed MAAPE (Modular Assembly Analysis of Protein Embeddings), a recently developed algorithm based on large protein language models (PLMs), to glean insights into the evolution process of KSs and CLFs. Our findings indicated the evolutionary history of KS and CLF domains from bacterial T2PKSs and identified a shared ancestral cluster (Cluster A), supporting a common origin. Despite structural homology, KSs and CLFs followed distinct evolutionary paths, shaped by coevolution and early horizontal gene transfer (HGT).

## Features
- PLM Embeddings: Utilization of ESM-2 (Evolutionary Scale Modeling) to extract high-dimensional protein features.
- MAAPE Algorithm: Application of the Modular Assembly Analysis of Protein Embeddings to reconstruct evolutionary trajectories.
- Functional Prediction: Integration of DeepFRI (Deep Functional Residue Identification) to correlate evolutionary clusters with GO molecular functions.
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

4. Builds a K-nearest neighbor (KNN) similarity network

5. Visualization\
Maps previously calculated directions and weights onto KNN edges\
Generates the final MAAPE (Molecular Assembly And Protein Engineering) network


## MAAPE workflow diagram
![graphical](https://github.com/user-attachments/assets/77610421-6d2d-44fb-bcb0-4944b8586c5a)

## MAAPE algorithm
![MAAPE算法示意图](https://github.com/user-attachments/assets/b36e147d-d28e-4784-9292-de9e3ae33e7a)

## 166 pairs of structurally analogous bacterial KS-CLF proteins and a Staphylococcus aureus FABF
<img width="4000" height="2250" alt="Figure-1" src="https://github.com/user-attachments/assets/cfe8b280-7c61-4bd3-99d6-e805b7855f83" />


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
git clone https://github.com/xiaoyuwang16/Qin-Lab_T2PKS-evolution.git
cd /content/Qin-Lab_T2PKS-evolution
pip install -r requirements.txt
```
## Contents and files

`/path/to/Qin-Lab_T2PKS-evolution/example` Contains the script & KS data used in our analysis:

+ _1_generate_embeddings.py       # Data processing pipeline code (6 files prefixed with _number_)
+  KS_all&outgroup.fasta           # the KS sequence, both KSα and KSβ
+  converted_thresholds_pca.npy    # A file containing sub-vector similarity thresholds corresponding to each window size. The methodology for calculating these thresholds is detailed in the MAAPE article.
+  order_index.txt                 # Index and protein classification (outgroup/KSα/KSβ) corresponding to `KS_all&outgroup.fasta`, used for color coding during visualization.
+  sequence_index.txt              # Index and protein names (ie. ksa_+ABXA-BE-24566B_13) corresponding to `KS_all&outgroup.fasta`.
+  DeepFRI function predictions.csv # Result file generating from DeepFRI.
+  Visual_Fig2a.py ...             # visualizing scripts for each figure in our paper.

 `/path/to/Qin-Lab_T2PKS-evolution/src ` Contains core MAAPE package modules.\
 `/path/to/Qin-Lab_T2PKS-evolution/constants.py ` \
 This file contains global parameters and path settings for the analysis pipeline. \
 **Before running the scripts, you must update the `BASE_DIR` to match your local environment.**\
All other scientific parameters (Window sizes, KNN, PCA) are pre-configured based on the methodology described in our manuscript for KS domain processing.
1. WINDOW_SIZES\
Used for sliding window analysis for sequence embedding, set multiple window sizes to capture sequence features at different scales.

2. COLOR_SCHEME\
Defines distinct color codes for different protein types\
Used for visualization

3. KNN Graph Parameters\
KNN_K = 20: Sets the number of nearest neighbors for each node\
KNN_THRESHOLD = 0.5: Defines the edge weight threshold

5. PCA Parameters\
PCA_COMPONENTS = 200: Sets the number of dimensions for dimensionality reduction, can't be too small for retaining key feature information

## Usage
```python
import os
import sys
maape_path = '/path/to/Qin-Lab_T2PKS-evolution' 
sys.path.append(maape_path)

import importlib.util

def import_file(file_path):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 1. Generate Embeddings
script = import_file('/path/to/Qin-Lab_T2PKS-evolution/example/_1_generate_embeddings.py')
generate_embeddings = script.main
generate_embeddings()
```
+ This step embedding FASTA sequences using ESM-2, followed by normalization and dimensionality reduction via PCA. The resulting file,  `/path/to/Qin-Lab_T2PKS-evolution/example/output/normalized_pca_embeddings.npy`, is stored in the output directory. This file has already been uploaded to the repository.
```
# 2. Generate Paths
script = import_file('/path/to/Qin-Lab_T2PKS-evolution/example/_2_generate_paths.py')
generate_paths = script.main
generate_paths()
```
+ This step generates 2 files: `/path/to/Qin-Lab_T2PKS-evolution/example/output/processed_paths.pkl` & `/path/to/Qin-Lab_T2PKS-evolution/example/output/search_results.pkl`.
```
# 3. Calculate Weights and Edges
script = import_file('/path/to/Qin-Lab_T2PKS-evolution/example/_3_calculate_weights_and_edges.py')
calculate_weights = script.main
calculate_weights()
```
+ This step will generate `/path/to/Qin-Lab_T2PKS-evolution/example/output/all_edges_data.pkl `.
```
# 4. Build and Analyze Graph
script = import_file('/path/to/MAAPE/Qin-Lab_T2PKS-evolution/_4_build_and_analyze_graph.py')
build_and_analyze = script.main
build_and_analyze()
```
+ This step generates `/path/to/Qin-Lab_T2PKS-evolution/example/output/knn_graph_edges.txt` & `/path/to/Qin-Lab_T2PKS-evolution/example/output/new_edge_weights_pca.pkl`

+ The amount of generated edges is huge in step 3, in Step 5 we screened the top 20% high-weighted edges for future analysis, generating `/path/to/Qin-Lab_T2PKS-evolution/example/output/selected_edges_data.pkl` & `/path/to/Qin-Lab_T2PKS-evolution/example/output/selected_edges.pkl`.
+ In step 6 we applied sparse encoding to the GO classifications predicted by DeepFRI. This converts the functional annotations of each sequence into vectors, facilitating the calculation of functional similarity.
+ You are now ready to run the Visual_*.py scripts.
<img width="4000" height="2250" alt="Figure-2" src="https://github.com/user-attachments/assets/754843d9-7730-4638-b05d-e72fe85f6057" />
<img width="4000" height="2250" alt="Figure-3" src="https://github.com/user-attachments/assets/7811d014-1b21-4d31-9858-3cee1e8cb900" />
<img width="4000" height="2250" alt="Figure-4" src="https://github.com/user-attachments/assets/a6c132a8-c230-4ea5-b9ce-573df69f1b67" />



