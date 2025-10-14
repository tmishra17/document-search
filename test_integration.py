#!/usr/bin/env python3
"""
Integration test for Chonkie chunking in the document search pipeline
"""
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from chonkie import SentenceChunker
import re

def preprocess_review(text: str) -> str:
    """Removes unwanted HTML from the provided text"""
    return re.sub(r'<br\s*?>', '\n', text, flags=re.IGNORECASE)

def test_integration():
    """Test the full pipeline with Chonkie chunking"""
    
    print("=" * 80)
    print("INTEGRATION TEST: Chonkie + Embeddings + Search")
    print("=" * 80)
    
    # Initialize components
    MODEL_NAME = "all-MiniLM-L6-v2"
    model = SentenceTransformer(MODEL_NAME)
    classifier = pipeline("sentiment-analysis")
    chunker = SentenceChunker(
        chunk_size=512,
        chunk_overlap=50,
        min_sentences_per_chunk=2
    )
    
    # Load sample data
    print("\n1. Loading sample reviews...")
    df = pd.read_csv("IMDB_Dataset.csv", nrows=10)
    print(f"   Loaded {len(df)} reviews")
    
    # Process reviews with chunking
    print("\n2. Processing reviews with Chonkie chunking...")
    chunk_rows = []
    chunk_texts = []
    
    for idx, row in df.iterrows():
        review_text = preprocess_review(row['review'])
        review_chunks = chunker.chunk(review_text)
        
        sentiment = classifier(review_text[:512])[0]['label']
        sentiment_score = classifier(review_text[:512])[0]['score']
        
        for chunk_obj in review_chunks:
            chunk_rows.append({
                'review': chunk_obj.text,
                'original_review': row['review'],
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'chunk_index': chunk_obj.start_index,
                'is_chunked': len(review_chunks) > 1
            })
            chunk_texts.append(chunk_obj.text)
    
    chunks_df = pd.DataFrame(chunk_rows)
    print(f"   Created {len(chunks_df)} chunks from {len(df)} reviews")
    print(f"   Average chunks per review: {len(chunks_df) / len(df):.2f}")
    
    # Create embeddings
    print("\n3. Creating embeddings for chunks...")
    text_embeddings = model.encode(chunk_texts, 
                                   convert_to_tensor=True,
                                   show_progress_bar=True)
    print(f"   Created embeddings with shape: {text_embeddings.shape}")
    
    # Test search
    print("\n4. Testing semantic search...")
    from sentence_transformers import util
    
    test_query = "amazing thriller movie"
    query_embedding = model.encode(test_query, convert_to_tensor=True)
    search_results = util.semantic_search(query_embedding, text_embeddings, top_k=3)[0]
    
    print(f"   Query: '{test_query}'")
    print(f"   Top 3 results:")
    
    for i, res in enumerate(search_results):
        idx = res['corpus_id']
        row_data = chunks_df.iloc[idx]
        print(f"\n   Result {i+1}:")
        print(f"     Score: {res['score']:.4f}")
        print(f"     Sentiment: {row_data['sentiment']}")
        print(f"     Is chunked: {row_data['is_chunked']}")
        print(f"     Chunk preview: {row_data['review'][:150]}...")
    
    print("\n" + "=" * 80)
    print("✅ Integration test completed successfully!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    test_integration()
