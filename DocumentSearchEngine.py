from docling.document_converter import DocumentConverter
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os
import pandas as pd
import re
from qdrant_client import QdrantClient, models
from transformers import pipeline

client = QdrantClient("localhost", port=6333)
classifier = pipeline("sentiment-analysis")
# need to query IMDB Kaggle database
def preprocess_review(text: str) -> str:
    """Removes unwanted HTML from the provided text

    Args:
        text (str): IMBD movie review

    Returns:
        str: review with removed HTML
    """
    return re.sub(r'<br\s*?>', '\n', text, flags=re.IGNORECASE)

# need a sentiment classifier so that score gets ranked by positive reviews
BATCH_SIZE = 1000
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_PATH = "/home/tmishra/my_space/document_project/review.pkl"
# Text Embedder
DB_PATH = "/home/tmishra/my_space/document_project/IMDB_Dataset.csv"
model = SentenceTransformer(MODEL_NAME)
@st.cache_data
def index_values() -> tuple[torch.Tensor, pd.DataFrame]:
    status = st.empty()
    if os.path.exists(EMBEDDING_PATH):
        status = st.info("Getting review embeddings...")
        with open(EMBEDDING_PATH, "rb") as pkl:
            text_embeddings, full_df = pickle.load(pkl)
        
        status.success("Successfully Loaded Embeddings!")
        status.empty()
        return text_embeddings, full_df
    else: 
        status.info("Loadinging Embeddings... may take a couple minutes")
        chunks = pd.read_csv("IMDB_Dataset.csv", chunksize=BATCH_SIZE)
        all_text_embeddings = []
        df_list = []
        for chunk in chunks:
            print(f"Applied Review {chunks['review']}")
            
            text_embeddings = model.encode(chunk['review'].tolist(), 
                                        batch_size=BATCH_SIZE,
                                        convert_to_tensor = True,
                                        convert_to_numpy = False
                                        )
            
            df_list.append(chunk)
            all_text_embeddings.append(text_embeddings)
        
        # Step 1: Combine all embeddings from chunks into one tensor
        text_embeddings = torch.cat(all_text_embeddings)
        
        # Step 2: Combine all DataFrame chunks into one complete DataFrame
        full_df = pd.concat(df_list, ignore_index=True)
        
        
        with open(EMBEDDING_PATH, "wb") as pkl:
            pickle.dump((text_embeddings, full_df), pkl)
        status.success("Successfully loaded embeddings")
        status.empty()
        return text_embeddings, full_df
def semantic_search(query: str, 
                    text_embeddings: torch.Tensor, 
                    threshold: float, 
                    max_results: int, 
                    chunks: pd.DataFrame) -> list:
    query_embedding = model.encode(query, convert_to_tensor=True)
    search_results = util.semantic_search(query_embedding, text_embeddings, top_k=int(max_results))[0]
    filtered_results = []
    for res in search_results:
        if res['score'] >= float(threshold):
            id = res['corpus_id']
            row_data = chunks.iloc[id].to_dict()
            filtered_results.append({"review": row_data['review'], "score": res['score']})
    return filtered_results

# how would I parse my database in docling, make individual indices of each one, and then 


def main():
    st.title("Describe a movie to search for something")

    text_embeddings, chunks = index_values()
    query = st.text_input("Enter Description here", placeholder="e.g. Good Alien Horror, Comedy")
    st.sidebar.header("⚙️ Search Settings")
    threshold=st.sidebar.slider("Similarity Threshold",
                      min_value = 0.0,
                      max_value = 1.0,
                      value = 0.2,
                      step = 0.05,
                      help="How relevant you want the documents to be"
                      )
    max_results = st.sidebar.slider("Max Results",
                      min_value = 1,
                      max_value = 50,
                      value = 9,
                      step = 1,
                      help= "Maximum number of results display for query")
    if st.button("Search 🔍"):
        results = semantic_search(query, 
                                  text_embeddings, 
                                  threshold, 
                                  max_results, 
                                  chunks)
        for res in results:
            st.markdown(f"**Score:** {res['score']:.3f}")
            st.markdown(res['review'])
            st.markdown("---")

if __name__ == "__main__":
    main()
