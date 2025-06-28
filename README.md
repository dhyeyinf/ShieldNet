# 🚨 ISCXIDS2012 - Network Intrusion Detection using Machine Learning

A comprehensive project that benchmarks multiple ML models to detect network intrusions using the ISCXIDS2012 dataset. The system processes day-wise traffic data and evaluates models like Random Forest, SVM, ExtraTrees, Gradient Boost, and more. The pipeline is fully automated and generates performance metrics for all models on all days.

---

## 📁 Project Structure

```bash
ISCXIDS2012-MASTER/
├── algorithms/              # Model implementations
├── data/                    # Dataset (original, CSV, split versions)
│   ├── original/            # Original CSVs from ISCXIDS2012
│   ├── CSV/                 # Cleaned data
│   └── split-CSV/           # 70-30 split data (train-test)
├── plotting/                # Plotting scripts (optional)
├── results/                 # Output metrics and result JSONs
│   ├── single/              # Results from single run per model
│   └── cv/                  # Results from cross-validation
├── runner-scripts/          # Automation scripts
├── ml.py                    # Entry point for training models
├── result_handling.py       # Handles and stores model results
├── extract_all_metrics.py   # Consolidates metrics from results
├── preproc.py               # Preprocessing and data splitting
├── reduction.py             # Feature selection/reduction
└── README.md                # Project readme
