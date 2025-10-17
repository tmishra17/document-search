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
from chonkie import SentenceChunker
# my only goal right now is to find the right bert model for sentiment analysis and see how 
# to integrate it into my code.
client = QdrantClient("localhost", port=6333)
# when I get the classifier score I need to see how it
# fit into my code, how about tomorrow I figure out how to
# do this tomorrow and come up with a plan
# maybe come up with pseudocode as well



# need a sentiment classifier so that score gets ranked by positive reviews
BATCH_SIZE = 1000
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_PATH = "/home/tmishra/my_space/document_project/review.pkl"
# Text Embedder
DB_PATH = "/home/tmishra/my_space/document_project/IMDB_Dataset.csv"
classifier = pipeline(
    "sentiment-analysis",
    # model="j-hartmann/sentiment-roberta-large-english-3-classes", # may need to train model
    device=-1  # Force CPU mode
)
chunker = SentenceChunker(chunk_size=400, chunk_overlap=50)
model = SentenceTransformer(MODEL_NAME)


# need to query IMDB Kaggle database
def preprocess_review(text: str) -> str:
    """Removes unwanted HTML from the provided text

    Args:
        text (str): IMBD movie review

    Returns:
        str: review with removed HTML
    """
    return re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)


def process_with_chonkie(df: pd.DataFrame) -> pd.DataFrame:
    """Process reviews with Chonkie - chunk long reviews into smaller pieces"""
    processed_rows = []
    chunked_count = 0
    
    for idx, row in df.iterrows():
        review_text = row['review']
        # Estimate tokens (1 token ≈ 4 characters)
        estimated_tokens = len(review_text) / 4

        if estimated_tokens > 450:  # Chunk if over 450 tokens
            # Chunk THIS individual review
            review_chunks = chunker(review_text)
            chunked_count += 1
                
            # Create a row for each chunk
            for i, chunk in enumerate(review_chunks):
                chunk_row = row.copy()
                chunk_row['review'] = chunk.text
                chunk_row['chunk_id'] = f"{idx}_{i}"
                chunk_row['original_id'] = idx
                chunk_row['is_chunked'] = True
                chunk_row['chunk_tokens'] = chunk.token_count
                processed_rows.append(chunk_row)
        
        else:
            # Keep short reviews unchanged
            row_copy = row.copy()
            row_copy['chunk_id'] = str(idx)
            row_copy['original_id'] = idx
            row_copy['is_chunked'] = False
            row_copy['chunk_tokens'] = int(estimated_tokens)
            processed_rows.append(row_copy)
    
    result_df = pd.DataFrame(processed_rows)
    st.info(f"📊 Chunked {chunked_count} long reviews into {len(result_df) - len(df) + chunked_count} total chunks")
    return result_df

def upload_to_qdrant(text_embeddings: torch.Tensor, df: pd.DataFrame, collection_name: str = "movie_reviews"):
    """Upload embeddings to Qdrant with proper error handling"""
    try:
        print("Qdrant already there")
        # Check if collection exists
        collection_info = client.get_collection(collection_name)
        if collection_info.points_count > 0:
            st.success(f"Qdrant collection already has {collection_info.points_count} points.")
            return True
    except Exception:
        # Create collection
        print("Creating Qdrant")
        st.info(f"Creating Qdrant collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=text_embeddings.shape[1],  # 384 for all-MiniLM-L6-v2
                distance=models.Distance.COSINE
            )
        )
    
    # Upload in batches
    st.info("Uploading to Qdrant...")
    upload_batch_size = 100
    points = []
    
    for i in range(len(df)):
        point = models.PointStruct(
            id=i,
            vector=text_embeddings[i].cpu().numpy().tolist(),
            payload={
                "review": df.iloc[i]['review'],
                "sentiment": df.iloc[i]['sentiment'],
                "sentiment_score": float(df.iloc[i]['sentiment_score']),
                "chunk_id": df.iloc[i]['chunk_id'],
                "is_chunked": bool(df.iloc[i]['is_chunked'])
            }
        )
        points.append(point)
        
        if len(points) == upload_batch_size:
            client.upsert(collection_name=collection_name, points=points)
            st.progress((i + 1) / len(df), text=f"Uploaded {i+1}/{len(df)} to Qdrant")
            points = []
    
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    st.success(f"Successfully uploaded {len(df)} chunks to Qdrant!")
    return True
        
    
@st.cache_data
def index_values() -> tuple[torch.Tensor, pd.DataFrame]:
    status = st.empty()
    if os.path.exists(EMBEDDING_PATH):
        status.info("Getting review embeddings...")
        with open(EMBEDDING_PATH, "rb") as pkl:
            text_embeddings, full_df = pickle.load(pkl)
        
        status.success("Successfully Loaded Embeddings!")
        status.empty()
        upload_to_qdrant(text_embeddings, full_df)
        return text_embeddings, full_df
    else: 
        status.info("Loadinging Embeddings... may take a couple minutes")
        chunks = pd.read_csv("IMDB_Dataset.csv", chunksize=BATCH_SIZE)
        all_text_embeddings = []
        df_list = []
        for chunk in chunks:
            # Fix 1: Clean the text first
            chunk['review'] = chunk['review'].apply(preprocess_review)
            chunk = process_with_chonkie(chunk)
            # Fix 2: Add sentiment analysis correctly
            chunk['sentiment'] = chunk['review'].apply(lambda x: classifier(x[:512])[0]['label'])
            chunk['sentiment_score'] = chunk['review'].apply(lambda x: classifier(x[:512])[0]['score'])
            
            # pass in sentiment and review so that we can analyze meaning
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
        upload_to_qdrant(text_embeddings, full_df)

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
    query_sentiment = classifier(query)[0]['label']
    query_embedding = model.encode(query, convert_to_tensor=True)
    search_results = util.semantic_search(query_embedding, text_embeddings, top_k=int(max_results))[0]
    
    filtered_results = []
    for res in search_results:
        if res['score'] >= float(threshold):
            id = res['corpus_id']
            row_data = chunks.iloc[id].to_dict()
            
            # Simple sentiment boost
            if row_data['sentiment'] == query_sentiment:
                filtered_results.append({
                    "review": row_data['review'], 
                    "score": res['score'],
                    "sentiment": row_data['sentiment'],
                    "sentiment_score": row_data['sentiment_score']
                })
    return sorted(filtered_results, key=lambda x: x['score'], reverse=True)

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
            
            st.markdown(f"**Sentiment:** {res['sentiment']}")
            if res['sentiment'] == "NEGATIVE":
                st.markdown(f"**Sentiment Score:** -{res['sentiment_score']:.3f}")
            else:
                st.markdown(f"**Sentiment Score:** {res['sentiment_score']:.3f}")
            st.markdown(res['review'])
            st.markdown("---")

if __name__ == "__main__":
    main()
