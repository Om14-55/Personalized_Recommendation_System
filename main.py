import streamlit as st
import pandas as pd
import pickle
from surprise import dump
from surprise import SVD

from math import isnan

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

# --- Load Models ---
@st.cache_resource
def load_models():
    _, svd_model = dump.load('svd_model.pkl')
    with open("content_data.pkl", "rb") as f:
        cosine_sim, indices = pickle.load(f)
    return svd_model, cosine_sim, indices

# --- Hybrid Prediction ---
def get_content_prediction(user_id, product_id, df, cosine_sim, indices):
    try:
        idx = indices[product_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        similar_indices = [i[0] for i in sim_scores]
        similar_ids = df.iloc[similar_indices]['product_id']

        user_ratings = df[(df['user_id'] == user_id) & (df['product_id'].isin(similar_ids))]['rating']
        return user_ratings.mean() if not user_ratings.empty else None
    except:
        return None

def hybrid_prediction(user_id, product_id, svd_model, cosine_sim, indices, df, alpha=0.5):
    content_pred = get_content_prediction(user_id, product_id, df, cosine_sim, indices)
    try:
        collab_pred = svd_model.predict(user_id, product_id).est
    except:
        collab_pred = None

    if content_pred and collab_pred:
        return alpha * content_pred + (1 - alpha) * collab_pred
    elif collab_pred:
        return collab_pred
    elif content_pred:
        return content_pred
    else:
        return "Not enough data to predict."

# --- Streamlit Interface ---
st.title(" Hybrid Product Recommender System")

df = load_data()
svd_model, cosine_sim, indices = load_models()

user_ids = df['user_id'].unique()
product_ids = df['product_id'].unique()

user_id = st.selectbox("Select a User ID", sorted(user_ids))
product_id = st.selectbox("Select a Product ID", sorted(product_ids))
alpha = st.slider("Alpha (blend weight)", 0.0, 1.0, 0.5, 0.1)

if st.button("Show Top 10 Recommendations"):
    # Products not yet rated by this user
    rated_products = df[df['user_id'] == user_id]['product_id'].unique()
    unrated_products = [pid for pid in product_ids if pid not in rated_products]

    recommendations = []

    for pid in unrated_products:
        pred = hybrid_prediction(user_id, pid, svd_model, cosine_sim, indices, df, alpha)
        if isinstance(pred, float) and not isnan(pred):
            recommendations.append((pid, round(pred, 2)))

    # Sort and display top 10
    top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    if top_recommendations:
        st.subheader(" Top 10 Recommended Products")
        for i, (pid, rating) in enumerate(top_recommendations, 1):
            st.write(f"{i}. **{pid}** - Predicted Rating: ⭐ {rating}")
    else:
        st.warning("Not enough data to generate recommendations.")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
