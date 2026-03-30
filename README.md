# Classifying Lithic 3D Point Clouds Using Deep Learning

## 📌 Purpose

This repository accompanies the manuscript:  
__Classifying Lithic 3D Point Clouds Using Deep Learning__  

It aims to:  
- Support reproducibility of results  
- Provide a reference implementation for 3D point cloud classification in archaeology  

## 📦 Repository Contents
- DGCNN notebook — PyTorch implementation of Dynamic Graph CNN  
- PointNet++ notebook — PyTorch implementation of PointNet++  
- Shared workflow:  
  Independent testing  
  Model evaluation  
  Critical point analysis  
  Analysis notebook — Generates figures and tables for the manuscript  

## 📁 Project Structure
### Code
Testing notebooks are located in the Test folder:  
Test/  
├── DGCNN_final/  
│   └── dgcnn_test.ipynb  
├── PointNet2_final/  
│   └── pointnet2_test.ipynb  

### Data
All required data are stored in the Data_share folder: 
Data_share/  
├── DGCNN/  
├── PointNet2/  
└── Independent_tests/  

### Analysis
Analysis notebook is stored in the Analysis folder:  
Analysis/  
├── analysis.ipynb   

## ⚙️ Output Directory Setup
For both notebooks, set:  
output_root_dir = <path_to_Data_share>/<ModelName>/Output/  
Where:  
ModelName = DGCNN or PointNet2  

Outputs are saved in:  
- DGCNN/Output/  
- PointNet2/Output/  

Includes:  
- Batch predictions  
- HTML visualizations  
- Critical point analysis results  

## ▶️ Run Control Settings
- RUN_INDEPENDENT_TEST = True      # Run independent tests  
- CHECK_ALREADY_RUN = True         # Skip completed tests  
- RERUN_EXISTING_TESTS = True      # Save reruns to a new folder  
- RUN_TAG = "rep_v1"               # Folder name (None = timestamp)  

### 🔁 Re-running Experiments
All experiments have already been executed.  

#### Option 1 — Reuse / overwrite results
- CHECK_ALREADY_RUN = True
- RERUN_EXISTING_TESTS = False
Skips completed runs
May overwrite outputs

#### Option 2 — Save to a new folder (recommended)
- RERUN_EXISTING_TESTS = True  
- RUN_TAG = "your_run_name"  
Creates a new output directory  
Preserves existing results  
Saves all test and analysis outputs  

### 🧩 Notes
Both notebooks share the same workflow and configuration  
Only the model architecture differs (DGCNN vs PointNet++)  
For reproducibility, use reruns with a custom RUN_TAG  

## 📊 Analysis
The Analysis notebook reproduces figures and tables from the manuscript.  
Set the root data path:  
- DATA_SHARE_DIR = "/path/to/Data_share"  

The notebook assumes the following structure:  
- POINTNET2_OUTPUT_ROOT = Data_share/PointNet2/Output
- DGCNN_OUTPUT_ROOT     = Data_share/DGCNN/Output

Key subdirectories:  
- independent_test_batches/ — test results  
- training/ — training outputs  

Ensure folder names remain unchanged so paths resolve correctly.  
