import streamlit as st
import re
import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from langchain.tools import Tool
import pickle
from sklearn.preprocessing import normalize

# Import SentenceTransformer and initialize it safely
try:
    from sentence_transformers import SentenceTransformer
    # Load SentenceTransformer (the same model used for training KMeans)
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded SentenceTransformer model: all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading SentenceTransformer: {e}")
    print(f"Error loading SentenceTransformer: {e}")
    sentence_model = None

# Insert KMeans model loading code
try:
    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    print("Loaded KMeans model from kmeans_model.pkl")
except Exception as e:
    st.error(f"Error loading KMeans model: {e}")
    print(f"Error loading KMeans model: {e}")
    kmeans_model = None

# Replace Ollama embedding functions with SentenceTransformer
def get_embedding(text):
    """Get embeddings using SentenceTransformer with the same model used for training KMeans"""
    try:
        # Get embeddings
        embedding = sentence_model.encode([text])[0]
        # Normalize the embedding to match training data preprocessing
        embedding = normalize(embedding.reshape(1, -1))[0]
        print(f"Debug - Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        return embedding
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        print(f"Error getting embedding: {e}")
        return None

def get_embeddings(sentences):
    """Get embeddings for multiple sentences"""
    embeddings = []
    for sentence in sentences:
        emb = get_embedding(sentence)
        if emb is not None:
            embeddings.append(emb)
    return embeddings

# --- Fine-Tuned Transformer Classifier ---
def load_classifier():
    return pipeline("text-classification", model="Dragneel/ticket-classification-v1")

classifier = load_classifier()

# User-friendly labels for classification models
id_to_label = {
    0: 'Billing Question', 
    1: 'Feature Request', 
    2: 'General Inquiry', 
    3: 'Technical Issue'
}

# --- Rule-Based Classifier Setup ---
categories = {
    'Billing Question': ['charge', 'payment', 'bill', 'subscription', 'refund'],
    'Feature Request': ['feature', 'add', 'improve', 'enhance', 'dark mode'],
    'General Inquiry': ['hours', 'schedule', 'information', 'details', 'inquiry'],
    'Technical Issue': ['crash', 'bug', 'error', 'issue', 'technical', 'upload']
}

def rule_based_classifier(ticket):
    scores = {cat: 0 for cat in categories}
    for cat, keywords in categories.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', ticket, re.IGNORECASE):
                scores[cat] += 1
    max_cat = max(scores, key=scores.get)
    return max_cat if scores[max_cat] > 0 else "General Inquiry"

# --- Clustering Classifier Setup ---
def clustering_classifier(ticket, kmeans_model):
    # Get embedding using the exact same approach used during training
    emb = get_embedding(ticket)
    if emb is None:
        return "Unknown"
    
    # Reshape for prediction (no need to convert dtype as SentenceTransformer 
    # should already return the correct format)
    ticket_emb = np.array(emb).reshape(1, -1)
    print(f"Debug - Embedding for prediction - shape: {ticket_emb.shape}, dtype: {ticket_emb.dtype}")
    
    # Use the KMeans model to predict the cluster
    cluster_pred = kmeans_model.predict(ticket_emb)
    
    # Map cluster numbers to meaningful labels based on user's specification
    cluster_labels = {
        0: "Technical Issue",
        1: "Feature Request",
        2: "Billing Question",
        3: "General Inquiry"
    }
    
    # Return a descriptive label based on the cluster prediction
    cluster_num = cluster_pred[0]
    if cluster_num in cluster_labels:
        return cluster_labels[cluster_num]
    return f"Cluster {cluster_num}"

# --- Unified Classification Function ---
def classify_ticket(ticket, method="fine-tuned"):
    if method == "fine-tuned":
        result = classifier(ticket)
        # Convert numeric label to human-readable label if needed
        try:
            label_id = int(result[0]['label'].split('_')[-1])
            label = id_to_label.get(label_id, result[0]['label'])
            return label, result[0]['score']
        except (ValueError, KeyError, IndexError):
            # If conversion fails, return the original label
            return result[0]['label'], result[0]['score']
    elif method == "rule-based":
        label = rule_based_classifier(ticket)
        return label, 1.0
    elif method == "clustering":
        label = clustering_classifier(ticket, kmeans_model)
        return label, 1.0
    else:
        return "Unknown method", 0.0

# Wrap the classifier in a LangChain tool.
ticket_classifier_tool = Tool(
    name="TicketClassifier",
    func=lambda ticket, method="fine-tuned": classify_ticket(ticket, method),
    description="Classify a support ticket using fine-tuned, rule-based, or clustering approaches."
)

# --- Streamlit UI ---
st.title("Ticket Classification with Ollama Embedding Model")
st.markdown("Select a classification method and enter your support ticket text below.")

ticket_input = st.text_area("Enter ticket text:")
method = st.selectbox("Select Classification Method:", ["fine-tuned", "rule-based", "clustering"])

if st.button("Classify Ticket"):
    if ticket_input.strip() == "":
        st.error("Please enter a ticket text.")
    else:
        label, confidence = ticket_classifier_tool.func(ticket_input, method=method)
        st.markdown(f"**Ticket:** {ticket_input}")
        st.markdown(f"**Predicted Category:** {label}")
        st.markdown(f"**Confidence:** {confidence:.4f}")
