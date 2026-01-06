# ISCXIDS2012 - Network Intrusion Detection using Machine Learning

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
├── run_all_model.sh         # Run every ML model on all 6 days
└── README.md                # Project readme
```

Technologies Used 
* Python 3.x
* `scikit-learn`
* `pandas`, `numpy`
* `matplotlib` (for optional plotting)
* Dataset: [ISCXIDS2012](https://www.unb.ca/cic/datasets/ids.html)

## Setup Instruction
1\. Clone the Repository
```bash
git clone https://github.com/dhyeyinf/ShieldNet.git
cd ShieldNet
```
2\. Install Dependencies
Make sure Python is installed (≥3.7), then:
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, you can manually install:
```bash
pip install scikit-learn pandas numpy matplotlib
```

## Dataset Structure
The dataset contains six days of network traffic:
| `-D` | Day       | Filename                 | Attack Scenario                 |
| ---- | --------- | ------------------------ | ------------------------------- |
| 0    | Monday    | TestbedMonJun14Flows.csv | HTTP DoS (Hulk)                 |
| 1    | Tuesday   | TestbedTueJun15Flows.csv | DDoS via IRC Botnet             |
| 2    | Wednesday | TestbedWedJun16Flows.csv | Brute Force                     |
| 3    | Thursday  | TestbedThuJun17Flows.csv | SSH Brute Force                 |
| 4    | Saturday  | TestbedSatJun19Flows.csv | Additional Brute Force Variants |
| 5    | Sunday    | TestbedSunJun20Flows.csv | Infiltration Attack             |

Each file is split into 70% training and 30% testing data during preprocessing.

### How to Run
1\. Run All Models on All Days
From the project root:
```bash
chmod +x runner-scripts/run_all_models.sh
./runner-scripts/run_all_models.sh
```
This will: 
* Run every ML model on all 6 days.
* Save JSON results in `results/single/<model_name>/`

2\. Extract Metrics into CSV
```bash
python extract_all_metrics.py
```
This will generate a consolidated CSV report with:
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Runtime
- Day (-D)
- Model name
## Available Models
The following models are evaluated in the project:
- K-Nearest Neighbors (`knn`)
- Nearest Centroid (`ncentroid`)
- Decision Tree (`dtree`)
- Linear SVM (`linsvc`)
- RBF SVM (`rbfsvc`)
- Random Forest (`rforest`)
- AdaBoost (`ada`)
- Bagging (`bag`)
- Logistic Regression (binlr)
- Quadratic Discriminant Analysis(`qda`)
- Linear Discriminant Analysis(`lda`)
- XGBoost (`xgboost`)
- Gradient Boost (`gradboost`)
- Extremely Randomised Trees (`extratree`)

Each model can also be run individually using:
```bash
python ml.py -D 0 -F rforest
```

Change `-D` to target day and `-F` to the model short-name.

## Plotting (Optional)
If needed later:
```bash
python plotting/plot_single_metrics.py -D 0 -F rforest
```
### Sample Output
After running, the results will be visible in:
```bash
results/single/<model_name>/<day>_<model>_Z_<%.json>
```
And the extracted final CSV report will summarize everything.
