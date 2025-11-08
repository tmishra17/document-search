from docling.document_converter import DocumentConverter
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import re
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import pipeline
from chonkie import SemanticChunker
# my only goal right now is to find the right bert model for sentiment analysis and see how 
# to integrate it into my code.
client = QdrantClient(url="http://0.0.0.0:6333")
# when I get the classifier score I need to see how it
# fit into my code, how about tomorrow I figure out how to
# do this tomorrow and come up with a plan
# maybe come up with pseudocode as well
# how do I make this a rag program
imdb = load_dataset("imdb")
def get_model():
    """Load model with proper device handling"""
    model = SentenceTransformer(MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to('cuda')
    # check if computer contains mps gpu, then check if it is available
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model = model.to('mps')
    return model
# need a sentiment classifier so that score gets ranked by positive reviews
BATCH_SIZE = 256
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_PATH = "/home/tmishra/my_space/document_project/review.pkl"
# Text Embedder
DB_PATH = "/home/tmishra/my_space/document_project/IMDB_Dataset.csv"
classifier = pipeline(
    "sentiment-analysis",
    model="j-hartmann/sentiment-roberta-large-english-3-classes", # may need to train model
    device=-1  # Force CPU mode
)
sem_chunker = SemanticChunker()
model = SentenceTransformer(MODEL_NAME)
model = get_model()

# def train_bert(epochs: int):
#     pass

def index_movie_data(collection_name: str = "movie_data"):
    status = st.empty()
    
    try:
        status.info("Getting movie data embeddings from qdrant...")
        client.get_collection(collection_name)
        status.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        status.empty()
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=368,
                distance=Distance.COSINE
            )
        )
    chunks_ratings = pd.read_csv("ratings.csv", chunksize=BATCH_SIZE)
    all_text_embeddings = []
    points = []
    for chunk in chunks_ratings:
        text_embeddings = model.encode(chunk.tolist(),
                                       convert_to_tensor=True,
                                       convert_to_numpy=False
                                    )
        all_text_embeddings.append(text_embeddings)
        point = PointStruct(
            id=chunk[''],
            payload= {
                "rating": chunk['rating'],
            }
        )
    

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
    for idx, row in df.iterrows():
        review_text = row['review']
        
        # Only chunk if long enough
        if len(review_text) > 30:
            sem_chunk = sem_chunker.chunk(review_text)
            
            # Create FLAT rows, not nested dicts
            for i, sc in enumerate(sem_chunk):
                chunk_row = row.copy()
                chunk_row['review'] = sc.text  
                chunk_row['chunk_id'] = f"{idx}_{i}"
                chunk_row['original_id'] = idx
                chunk_row['is_chunked'] = True
                chunk_row['start_char'] = sc.start_index
                chunk_row['end_char'] = sc.end_index
                chunk_row['parent_review'] = review_text
                chunk_row['chunk_index'] = i
                chunk_row['total_chunks'] = len(sem_chunk)
                processed_rows.append(chunk_row)
        else:
            # Keep short reviews as-is
            row_copy = row.copy()
            row_copy['chunk_id'] = str(idx)
            row_copy['original_id'] = idx
            row_copy['is_chunked'] = False
            processed_rows.append(row_copy)
    
    result_df = pd.DataFrame(processed_rows)
    st.info(f"📊 Created {len(result_df)} chunks from {len(df)} reviews")
    return result_df
  
def get_sentiment_info(text: str) -> tuple:
    """Get both sentiment label and score in one inference call"""
    # makes processing embeddings more efficient
    result = classifier(text[:512])[0]
    return result['label'], result['score']

@st.cache_data
def index_values(collection_name: str = "movie_reviews") -> tuple[torch.Tensor, pd.DataFrame]:
    status = st.empty()
    try:
        status.info("Getting review embeddings from qdrant...")
        client.get_collection(collection_name)
        status.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        status.empty()
    except Exception:
        st.error(f"Collection '{collection_name}' does not exist. Creating now.")
        status.info("Creating Embeddings... may take a couple minutes")
        chunks = pd.read_csv(DB_PATH, chunksize=BATCH_SIZE)
        all_text_embeddings = []
        for chunk in chunks:
            # Clean the text first
            chunk['review'] = chunk['review'].apply(preprocess_review)
            # chunk = process_with_chonkie(chunk)
            
            # Add sentiment analysis correctly
            chunk[['sentiment', 'sentiment_score']] = chunk['review'].apply(
                lambda x: pd.Series(get_sentiment_info(x))
            )   
            
            # pass in sentiment and review so that we can analyze meaning
            text_embeddings = model.encode(chunk['review'].tolist(), 
                                        batch_size=BATCH_SIZE,
                                        convert_to_tensor = True,
                                        convert_to_numpy = False
                                    )
            
            all_text_embeddings.append(text_embeddings)
            
        # Combine all embeddings from chunks into one tensor
        text_embeddings = torch.cat(all_text_embeddings)
        
        # Combine all DataFrame chunks into one complete DataFrame
        full_df = pd.concat(chunks, ignore_index=True)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=text_embeddings.shape[1],
                distance=Distance.COSINE
            )
        )
        
        
        st.info("Uploading to Qdrant...")
        upload_batch_size = 100
        points = []
        
        for i in range(len(full_df)):
            point = PointStruct(
                id=i,
                vector=text_embeddings[i].cpu().numpy().tolist(),
                payload={
                    "review": full_df.iloc[i].get("review", full_df.iloc[i]['review']),
                    "sentiment": full_df.iloc[i]['sentiment'],
                    "sentiment_score": float(full_df.iloc[i]['sentiment_score']),
                    # "chunk_id": full_df.iloc[i]['chunk_id']
                    # "is_chunked": bool(full_df.iloc[i]['is_chunked']),
                    # "chunk_index": int(full_df.iloc[i].get('chunk_index', 0)),
                }
            )
            points.append(point)
            
            if len(points) == upload_batch_size:
                client.upsert(collection_name=collection_name, points=points)
                st.progress((i + 1) / len(full_df), text=f"Uploaded {i+1}/{len(full_df)} to Qdrant")
                points = []
        
        if points:
            client.upsert(collection_name=collection_name, points=points)
        
        client.get_collection(collection_name)
        st.success(f"Successfully uploaded {len(full_df)} chunks to Qdrant!")
  
# stand out from google by making searches actually relevant to the user's query

def fetch_internet_reviews(movie_title: str, imdb_id: str) -> list:
    """Fetch reviews from internet (Google, IMDb)"""
    reviews = []
    
    # For now, return placeholder
    # In production, you'd use:
    # - IMDb API (requires key)
    # - Google Custom Search API
    # - Web scraping libraries
    
    reviews.append({
        "source": "IMDb",
        "rating": 8.5,
        "text": "Great movie! Highly recommended.",
        "author": "User123"
    })
    
    reviews.append({
        "source": "IMDb",
        "rating": 9.0,
        "text": "Amazing plot and cinematography!",
        "author": "MovieFan456"
    })
    
    return reviews  
  
  
def semantic_search(query: str, 
                    threshold: float, 
                    max_results: int, 
                    collection_name: str = "movie_reviews"
                ) -> list:
    # only need the first couple of lines to get the sentiment of the query
    query_sentiment = classifier(query[:512])[0]['label']
    query_embeddings = model.encode(query, convert_to_tensor=True)
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embeddings.cpu().numpy().tolist(),
        limit=max_results,
        score_threshold=threshold
    )
    filtered_results = []
    for res in search_results:
        payload = res.payload
        # Simple sentiment boost
        if payload['sentiment'] == query_sentiment:
            filtered_results.append({
                "review": payload['parent_chunk'], 
                "score": res.score,
                "sentiment": payload['sentiment'],
                "sentiment_score": payload['sentiment_score']
            })
    return sorted(filtered_results, key=lambda x: x['score'], reverse=True)

# how would I parse my database in docling, make individual indices of each one, and then 


def main():
    st.title("Describe a movie to search for something")

    index_values()
    st.sidebar.header("⚙️ Search Settings")
    threshold = st.sidebar.slider("Similarity Threshold",
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
    query = st.text_input("Enter Description here", placeholder="e.g. Good Alien Horror, Comedy")
    if st.button("Search 🔍") and query:
        st.empty()
        results = semantic_search(query, 
                                threshold, 
                                max_results
                                ) 
        for res in results:
            st.markdown(f"**Score:** {res['score']:.3f}")
            
            st.markdown(f"**Sentiment:** {res['sentiment']}")
            if res['sentiment'] == "NEGATIVE":
                st.markdown(f"**Sentiment Score:** -{res['sentiment_score']:.3f}")
            else:
                st.markdown(f"**Sentiment Score:** {res['sentiment_score']:.3f}")
            
            st.markdown(res['review'])
            st.markdown("---")
    
    else:
        st.error("please enter text to search")
       


if __name__ == "__main__":
    main()