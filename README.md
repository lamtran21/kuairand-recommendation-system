# kuairand-recommendation-system

## Collaborators
- Lam Tran
- Minhae Park
- Minhao Zhang
- Diana Chen
- Steven Si
- Akshaj Chandwani
- Khalil He

## Project Goals
This project evaluates several collaborative filtering approaches using the KuaiRand-Pure dataset, a large-scale dataset containing implicit user interaction signals from a short-video platform. 

We compare multiple recommendation methods, including popularity-based recommendations, Item-based Collaborative Filtering (ItemCF), Singular Value Decomposition (SVD), and Alternating Least Squares (ALS). In addition, we implement a two-stage recommendation pipeline in which ALS performs candidate retrieval and a Gradient Boosting model re-ranks the candidate set.

## Repository Structure
kuairand-recommendation_system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/                ---code to train models + predict
│   ├── popularity.py
│   ├── itemcf.py
│   ├── als.py
│   └── svd.py
│
├── experiments/       --call models + evaluation
│   ├── run_popularity.py
│   ├── run_itemcf.py
│   ├── run_als.py
│   └── run_svd.py
│
├── evaluation/
│   └── metrics.py      -- model agnostic metric calculation
│
├── notebooks/
│   └── eda.ipynb
│
├── outputs/
│   ├── models/   --pickled files
│   └── results/   --charts, graphs, slides, 10 page report
│
├── requirements.txt
└── README.md
