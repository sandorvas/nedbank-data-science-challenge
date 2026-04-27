Nedbank Data Challenge

Solution Summary

End-to-end machine learning pipeline using RandomForest regression on customer demographic data with log-transformed predictions to align with evaluation metric behavior.

Architecture Thinking

This solution follows a simple but robust data pipeline design:

1. Data Ingestion Layer
    * Load training, test, and demographic datasets
    * Ensure consistent schema alignment
2. Feature Engineering Layer
    * Transform BirthDate → Age (domain-relevant feature)
    * Apply one-hot encoding for categorical variables
    * Align train/test feature space to guarantee consistency
3. Modeling Layer
    * RandomForest chosen for stability and low overfitting risk
    * Avoided complex transformations to preserve prediction reliability
4. Prediction Layer
    * Raw predictions converted using log1p transformation
    * Ensures compatibility with competition evaluation behavior (RMSLE)
5. Output Layer
    * Submission strictly aligned with SampleSubmission format
    * Preserves row order and ID integrity

Key Engineering Decisions

* Prioritized pipeline correctness over model complexity
* Avoided overfitting by keeping model simple and stable
* Ensured strict alignment between features and submission format
* Identified and resolved multiple failure modes:
    * datatype inconsistencies
    * feature misalignment
    * transformation errors
    * submission structure issues

Final Score

RMSLE: ~1.075

Run

python final_model.py

Notes for Interview / Discussion

* Demonstrates ability to debug real-world ML pipelines under uncertainty
* Shows understanding of evaluation metrics and their practical implications
* Highlights importance of data integrity and transformation consistency
* Emphasizes engineering discipline over blind model optimization

Future Improvements

* Replace RandomForest with CatBoost for better categorical handling
* Introduce cross-validation for more reliable model evaluation
* Explore advanced feature encoding techniques (target/frequency encoding)
* Incorporate financial features once properly validated
