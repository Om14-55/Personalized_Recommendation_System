# Personalized Hybrid Recommendation System

A machine learning-based hybrid recommendation engine built using the Amazon product dataset. It intelligently combines collaborative filtering and content-based methods to deliver personalized product suggestions. The system is deployed with an interactive **Streamlit dashboard** for real-time use.

---

## Problem Statement

Build a smart product recommendation system that:
- Understands user behavior from historical review data.
- Learns product similarities from metadata using TF-IDF.
- Blends both collaborative and content-based recommendations using a tunable alpha parameter.
- Provides an intuitive UI to test, prioritize, and explore predictions.

---

## Dataset

We used the **Amazon Product Reviews Dataset**, which contains:
- User IDs, Product IDs
- Ratings and review text
- Product metadata (e.g., titles, categories)

---

## Models Used

| Technique               | Algorithms             |
|------------------------|------------------------|
| Collaborative Filtering| User-Based, Item-Based |
| Matrix Factorization   | SVD, NMF (Surprise)    |
| Content-Based Filtering| TF-IDF on metadata     |
| Hybrid Approach        | Blended CF + CBF       |

---

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Precision@K
- Predicted Ratings for Top-N Recommendations

---

## Streamlit Dashboard

> Below is a screenshot of the **deployed dashboard** built using Streamlit:

![Dashboard Screenshot](./Screenshot%202025-07-08%20191245.png)

- Select a user and a product ID
- Adjust the `Alpha` slider to blend content & collaborative recommendations
- View top 10 predicted products instantly!

---

## How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hybrid-recommendation-system.git
cd hybrid-recommendation-system
