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



def chunk_text(text: int, chunk_size: int = 500, overlap: int = 100) ->list:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

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

def store_embeddings_in_qdrant(embeddings: torch.Tensor, df: pd.DataFrame, status_widget):
    """
    Store matryoshka embeddings and metadata in Qdrant vector database with sentiment analysis.
    
    This function does the following:
    1. Creates a Qdrant collection for storing movie review embeddings
    2. Processes each review with sentiment analysis
    3. Stores embeddings along with rich metadata (review text, sentiment, etc.)
    4. Uploads everything to Qdrant in efficient batches
    
    Args:
        embeddings: PyTorch tensor containing all review embeddings
        df: DataFrame containing the review text and original sentiment labels
        status_widget: Streamlit widget for showing progress updates
    """
    collection_name = "movie_reviews"
    
    # Step 1: Check if collection already exists and recreate it for fresh data
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        print(f"Collection '{collection_name}' doesn't exist yet")
    
    # Step 2: Create new collection with proper vector configuration
    vector_size = embeddings.shape[1]  # Get embedding dimension (e.g., 1024 for matryoshka)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,           # Matryoshka embedding dimension
            distance=models.Distance.COSINE  # Use cosine similarity for semantic search
        )
    )
    status_widget.info(f"Created Qdrant collection with {vector_size}D vectors")
    
    # Step 3: Convert PyTorch tensors to Python lists (required by Qdrant)
    embeddings_list = embeddings.cpu().numpy().tolist()
    
    # Step 4: Process each review and create Qdrant points with sentiment analysis
    points = []
    total_reviews = len(df)
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Clean HTML tags from review text
        clean_review = preprocess_review(row['review'])
        
        # Perform sentiment analysis on cleaned text
        # Note: We truncate to 512 chars because most sentiment models have token limits
        truncated_text = clean_review[:512] if len(clean_review) > 512 else clean_review
        sentiment_result = classifier(truncated_text)
        
        # Extract sentiment predictions
        predicted_sentiment = sentiment_result[0]['label']  # POSITIVE or NEGATIVE
        sentiment_confidence = sentiment_result[0]['score']  # Confidence score 0-1
        
        # Create Qdrant point with embedding and rich metadata
        point = models.PointStruct(
            id=i,  # Unique ID for each review
            vector=embeddings_list[i],  # The matryoshka embedding vector
            payload={
                # Core data
                "review": clean_review,
                "original_sentiment": row['sentiment'],  # Ground truth from dataset
                
                # AI-predicted sentiment analysis
                "predicted_sentiment": predicted_sentiment,
                "sentiment_confidence": sentiment_confidence,
                
                # Additional metadata for filtering and analysis
                "review_length": len(clean_review),
                "word_count": len(clean_review.split()),
                
                # Sentiment agreement (does AI prediction match ground truth?)
                "sentiment_match": (predicted_sentiment.lower() == row['sentiment'].lower())
            }
        )
        points.append(point)
        
        # Show progress every 1000 reviews
        if i % 1000 == 0:
            progress = (i / total_reviews) * 100
            status_widget.info(f"Processing reviews with sentiment analysis: {progress:.1f}%")
    
    # Step 5: Upload to Qdrant in batches (more efficient than one-by-one)
    batch_size = 100  # Qdrant recommends smaller batches for stability
    total_batches = (len(points) + batch_size - 1) // batch_size
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        
        batch_num = (i // batch_size) + 1
        status_widget.info(f"Uploading to Qdrant: Batch {batch_num}/{total_batches}")
    
    print(f"✅ Successfully stored {len(points)} reviews with embeddings and sentiment analysis in Qdrant")

def search_with_qdrant(query: str, max_results: int = 10, sentiment_filter: str = None):
    """
    Search for similar movie reviews using Qdrant vector database.
    
    This provides several advantages over in-memory search:
    1. Much faster similarity search (optimized vector indices)
    2. Rich filtering capabilities (by sentiment, length, etc.)
    3. Scalable to millions of reviews
    4. Persistent storage (survives app restarts)
    
    Args:
        query: User's search query (e.g., "funny alien movie")
        max_results: Maximum number of results to return
        sentiment_filter: Optional filter ("POSITIVE", "NEGATIVE", or None for both)
    
    Returns:
        List of search results with reviews, scores, and metadata
    """
    # Step 1: Create embedding for the user's query using same model
    query_embedding = matryoshka_model.encode([query])
    
    # Step 2: Build filter conditions (optional)
    search_filter = None
    if sentiment_filter:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="predicted_sentiment",
                    match=models.MatchValue(value=sentiment_filter)
                )
            ]
        )
    
    # Step 3: Search Qdrant for most similar reviews
    search_results = client.search(
        collection_name="movie_reviews",
        query_vector=query_embedding[0].tolist(),
        query_filter=search_filter,
        limit=max_results,
        with_payload=True,  # Include all metadata in results
        score_threshold=0.1  # Only return reasonably similar results
    )
    
    # Step 4: Format results for display
    results = []
    for hit in search_results:
        results.append({
            # Core search data
            "review": hit.payload["review"],
            "similarity_score": hit.score,
            
            # Sentiment analysis
            "original_sentiment": hit.payload["original_sentiment"],
            "predicted_sentiment": hit.payload["predicted_sentiment"],
            "sentiment_confidence": hit.payload["sentiment_confidence"],
            "sentiment_match": hit.payload["sentiment_match"],
            
            # Review metadata
            "review_length": hit.payload["review_length"],
            "word_count": hit.payload["word_count"]
        })
    
    return results

def check_qdrant_status():
    """
    Check if Qdrant is running and has our movie reviews collection.
    This helps with error handling and user feedback.
    """
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "movie_reviews" in collection_names:
            collection_info = client.get_collection("movie_reviews")
            return {
                "status": "ready",
                "message": f"Found {collection_info.points_count} reviews in Qdrant",
                "count": collection_info.points_count
            }
        else:
            return {
                "status": "no_collection",
                "message": "Qdrant is running but no movie reviews found",
                "count": 0
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cannot connect to Qdrant: {str(e)}",
            "count": 0
        }

# Configuration and setup
BATCH_SIZE = 1000
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_PATH = "/home/tmishra/my_space/document_project/review.pkl"
# Text Embedder
DB_PATH = "/home/tmishra/my_space/document_project/IMDB_Dataset.csv"
matryoshka_model = SentenceTransformer(MODEL_NAME)
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
            
            text_embeddings = matryoshka_model.encode(chunk['review'].tolist(), 
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
        
        # Step 3: Store embeddings in Qdrant vector database with sentiment analysis
        store_embeddings_in_qdrant(text_embeddings, full_df, status)
        
        with open(EMBEDDING_PATH, "wb") as pkl:
            pickle.dump((text_embeddings, full_df), pkl)
        status.success("Successfully loaded embeddings")
        status.empty()
        return text_embeddings, full_df
def semantic_search(query: str, 
                    text_embeddings: torch.Tensor, 
                    threshold: float, 
                    max_results: int, 
                    chunks: pd.DataFrame,
                    sentiment_filter: str = None) -> list:
    """
    Enhanced semantic search using Qdrant vector database.
    
    This function now uses Qdrant instead of in-memory search for several benefits:
    1. Faster search performance (especially for large datasets)
    2. Advanced filtering capabilities (by sentiment, review length, etc.)
    3. Better scalability and memory efficiency
    4. Persistent storage that survives app restarts
    
    Args:
        query: User's search query
        text_embeddings: Legacy parameter (kept for compatibility)
        threshold: Minimum similarity score (0-1)
        max_results: Maximum number of results
        chunks: Legacy parameter (kept for compatibility)
        sentiment_filter: Optional sentiment filter ("POSITIVE" or "NEGATIVE")
    
    Returns:
        List of filtered search results with enhanced metadata
    """
    try:
        # Use Qdrant for the actual search (much faster and more scalable)
        qdrant_results = search_with_qdrant(query, max_results * 2, sentiment_filter)  # Get extra results for filtering
        
        # Apply similarity threshold filtering
        filtered_results = []
        for result in qdrant_results:
            if result['similarity_score'] >= float(threshold):
                # Format result for compatibility with existing UI
                formatted_result = {
                    "review": result['review'],
                    "score": result['similarity_score'],
                    "original_sentiment": result['original_sentiment'],
                    "predicted_sentiment": result['predicted_sentiment'],
                    "sentiment_confidence": result['sentiment_confidence'],
                    "sentiment_match": result['sentiment_match'],
                    "word_count": result['word_count']
                }
                filtered_results.append(formatted_result)
                
                # Stop when we have enough results
                if len(filtered_results) >= max_results:
                    break
        
        return filtered_results
    
    except Exception as e:
        # Fallback to original in-memory search if Qdrant fails
        st.error(f"Qdrant search failed: {e}")
        st.info("Falling back to in-memory search...")
        
        query_embedding = matryoshka_model.encode(query, convert_to_tensor=True)
        search_results = util.semantic_search(query_embedding, text_embeddings, top_k=max_results)[0]
        
        filtered_results = []
        for res in search_results:
            if res['score'] >= float(threshold):
                id = res['corpus_id']
                row_data = chunks.iloc[id].to_dict()
                filtered_results.append({
                    "review": row_data['review'], 
                    "score": res['score'],
                    "original_sentiment": row_data.get('sentiment', 'unknown'),
                    "predicted_sentiment": "unknown",
                    "sentiment_confidence": 0.0,
                    "sentiment_match": False,
                    "word_count": len(row_data['review'].split())
                })
        
        return filtered_results

# how would I parse my database in docling, make individual indices of each one, and then 


def main():
    st.title("🎬 AI-Powered Movie Review Search")
    st.markdown("*Search through movie reviews using semantic similarity and sentiment analysis*")
    
    # Check Qdrant status and display info
    qdrant_status = check_qdrant_status()
    if qdrant_status["status"] == "ready":
        st.success(f"✅ {qdrant_status['message']}")
    elif qdrant_status["status"] == "no_collection":
        st.warning(f"⚠️ {qdrant_status['message']} - Run setup first!")
    else:
        st.error(f"❌ {qdrant_status['message']}")
        st.info("Make sure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`")

    # Initialize embeddings (this will create Qdrant collection on first run)
    text_embeddings, chunks = index_values()
    
    # Main search interface
    query = st.text_input(
        "🔍 Describe what kind of movie reviews you're looking for:",
        placeholder="e.g. funny alien comedy, scary horror movie, romantic drama"
    )
    
    # Advanced search settings in sidebar
    st.sidebar.header("⚙️ Search Settings")
    
    # Similarity threshold
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Higher values = more relevant but fewer results"
    )
    
    # Number of results
    max_results = st.sidebar.slider(
        "Max Results",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Maximum number of reviews to display"
    )
    
    # Sentiment filtering (new feature!)
    sentiment_filter = st.sidebar.selectbox(
        "Filter by Sentiment",
        options=[None, "POSITIVE", "NEGATIVE"],
        format_func=lambda x: "All Reviews" if x is None else f"{x.title()} Reviews Only",
        help="Filter results by predicted sentiment"
    )
    
    # Display search metrics
    if qdrant_status["status"] == "ready":
        st.sidebar.metric("Total Reviews", f"{qdrant_status['count']:,}")
    
    # Perform search when button clicked
    if st.button("🔍 Search Reviews", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query!")
            return
        
        with st.spinner("Searching through movie reviews..."):
            results = semantic_search(
                query=query,
                text_embeddings=text_embeddings,
                threshold=threshold,
                max_results=max_results,
                chunks=chunks,
                sentiment_filter=sentiment_filter
            )
        
        if not results:
            st.warning(f"No reviews found with similarity >= {threshold:.2f}")
            st.info("Try lowering the similarity threshold or changing your search terms")
            return
        
        # Display results with enhanced information
        st.subheader(f"Found {len(results)} matching reviews:")
        
        for i, res in enumerate(results, 1):
            with st.expander(f"Review #{i} - Similarity: {res['score']:.3f}", expanded=(i <= 3)):
                
                # Metadata row with sentiment analysis
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Similarity", f"{res['score']:.3f}")
                with col2:
                    sentiment = res['predicted_sentiment']
                    confidence = res['sentiment_confidence']
                    st.metric("AI Sentiment", f"{sentiment}", 
                             delta=f"{confidence:.2f} confidence")
                with col3:
                    st.metric("Original Label", res['original_sentiment'])
                with col4:
                    match_emoji = "✅" if res['sentiment_match'] else "❌"
                    st.metric("AI Accuracy", match_emoji)
                
                # Word count info
                st.caption(f"📝 {res['word_count']} words")
                
                # The actual review text
                st.markdown("**Review:**")
                st.markdown(res['review'])
                
                # Add some spacing
                st.markdown("---")

if __name__ == "__main__":
    main()
