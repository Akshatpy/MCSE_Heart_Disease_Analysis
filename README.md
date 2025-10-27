# Heart Disease Analysis

Reference Kaggle Dataset : https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog
Brief: A small analysis script that loads the "Heart_disease_statlog.csv" dataset, introduces and fills artificial missing values, generates descriptive statistics, visualizations, a normality check, a confidence interval and hypothesis test for cholesterol, and fits a simple linear regression model.

Prerequisites
- Python 3.8+
- Recommended packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn [requirements.txt]

Quick install
- (optional) Create and activate a virtual environment
- Install dependencies:
  pip install -r requirements.txt 

Project structure (relevant)
- heart/main2.py        — main analysis script
- Heart_disease_statlog.csv — expected dataset (place next to main2.py or run from that folder)
- Generated outputs (saved by the script):
  - chol_histogram.png
  - chol_by_target_boxplot.png
  - age_vs_thalach_scatter.png
  - correlation_heatmap.png
  - chol_qq_plot.png
  - regression_actual_vs_pred.png

How to run
1. Ensure the CSV dataset `Heart_disease_statlog.csv` is in the same directory as `main2.py` (c:\Users\Akshat\Desktop\OrangeMath\heart\) or run the script from that folder.
2. From a command prompt:
   cd c:\Users\Akshat\Desktop\OrangeMath\heart
   python main2.py
