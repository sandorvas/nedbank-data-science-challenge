🧠 Nedbank Data Science Challenge

📌 Overview

This repository contains my solution for the Nedbank Data Science Challenge.

The objective is to build a predictive model using customer, transaction, and financial data to generate accurate predictions for the provided test dataset.

⸻

⚙️ Approach

The solution follows a structured data science workflow:

1. Data Integration
    * Merged core datasets using UniqueID
    * Combined demographic and financial feature sets
2. Feature Engineering
    * Used pre-engineered financial and demographic features
    * Ensured alignment between training and test datasets
3. Modeling
    * Trained a classification model using scikit-learn
    * Focused on simplicity, stability, and reproducibility
4. Evaluation
    * Generated predictions in the required submission format
    * Verified output against competition requirements

⸻

📂 Repository Structure

* StarterNotebook.ipynb — main notebook with full workflow
* evaluate.py — evaluation script
* .gitignore — excludes data and artifacts

⸻

▶️ How to Run

1. Install dependencies:

pip install pandas scikit-learn

2. Run the notebook:

jupyter notebook StarterNotebook.ipynb

⸻

🧩 Notes

* Data files are excluded from the repository per competition rules
* The solution is fully reproducible given the original datasets

⸻

🚀 Author

Sandor Vas
