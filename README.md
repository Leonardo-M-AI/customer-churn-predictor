# Customer Churn Predictor

Predicting whether telecom customers will cancel their subscription, using account and behavioural data. Comparing two machine learning approaches and translating model performance into a concrete business value estimate.

## Project Structure
customer-churn-predictor/

├── notebook/churn_predictor.ipynb

├── requirements.txt

└── .gitignore
## Pipeline

| Step | Description |
|------|-------------|
| Data Loading | 7,043 customer records via the IBM Telco Customer Churn dataset |
| EDA | Class balance, missing values, tenure and contract analysis |
| Preprocessing | Categorical encoding, feature scaling, train/test split with stratification |
| Models | Logistic Regression and Random Forest, both with class weighting |
| Evaluation | Accuracy, ROC-AUC, Classification Report, Confusion Matrix |
| Interpretation | Feature importance ranking to identify churn drivers |
| Business Translation | Converting model output into estimated revenue impact |

## Models

**Logistic Regression**: class-weighted to handle imbalance, scaled features, baseline interpretable model

**Random Forest**: 200 estimators, max depth 10, class-weighted, used for feature importance analysis

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 72.6% | 0.834 |
| Random Forest | 77.4% | 0.833 |

Random Forest correctly identifies **70.9% of customers who actually churned** in the test set, with comparable discriminative power (ROC-AUC) to Logistic Regression but better overall accuracy.

## Top Churn Drivers

Contract type, tenure, total charges, and monthly charges are the strongest predictors of churn, with contract type alone accounting for nearly 18% of the model's decision-making weight. Customers on month-to-month contracts churn at a substantially higher rate than those on annual or two-year plans.

## Quick Start

```bash
git clone https://github.com/Leonardo-M-AI/customer-churn-predictor.git
pip install -r requirements.txt
jupyter notebook notebook/churn_predictor.ipynb
```

## Tech Stack
`Python` `scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

## License
MIT

## Business Impact

On the test set alone, the model correctly flags customers responsible for an estimated $206,000 in annual recurring revenue at risk of being lost. Applied at scale, this kind of model allows a retention team to prioritise outreach toward the customers most likely to leave, rather than spreading limited budget across the entire customer base. Combined with the feature importance analysis, it also gives a clear, actionable signal: contract structure and tenure are the levers most worth addressing in a retention strategy.

## Limitations

The dataset is a single snapshot in time and does not capture seasonal or trend effects. Class imbalance (26.5% churn rate) means precision on the churned class remains moderate (56%) at the default threshold, so a portion of flagged customers will not actually churn. Lowering the decision threshold to 0.3 increases recall to roughly 88% (catching nearly all at-risk customers) at the cost of precision dropping to around 40%, a tradeoff that favours recall when the cost of losing a customer outweighs the cost of a wasted retention offer.

## What I Would Do Next

- Test gradient boosting models (XGBoost, LightGBM) for a potential accuracy gain
- Tune the decision threshold based on the actual cost of a retention offer versus the lifetime value of a retained customer
- Deploy the model as a REST API to score customers in real time
- A/B test a retention offer on the customers flagged as high-risk to validate real-world impact
