#!/usr/bin/env python3
"""
Test script to verify Chonkie chunking implementation
"""
import pandas as pd
from chonkie import SentenceChunker
import re

def preprocess_review(text: str) -> str:
    """Removes unwanted HTML from the provided text"""
    return re.sub(r'<br\s*?>', '\n', text, flags=re.IGNORECASE)

def test_chonkie_chunking():
    """Test Chonkie chunking on sample reviews"""
    
    # Initialize chunker with same settings as main app
    chunker = SentenceChunker(
        chunk_size=512,
        chunk_overlap=50,
        min_sentences_per_chunk=2
    )
    
    # Load sample reviews
    df = pd.read_csv("IMDB_Dataset.csv", nrows=5)
    
    print("=" * 80)
    print("CHONKIE CHUNKING TEST")
    print("=" * 80)
    
    total_chunks = 0
    
    for idx, row in df.iterrows():
        review_text = preprocess_review(row['review'])
        review_chunks = chunker.chunk(review_text)
        
        total_chunks += len(review_chunks)
        
        print(f"\n--- Review {idx + 1} ---")
        print(f"Original length: {len(review_text)} chars")
        print(f"Number of chunks: {len(review_chunks)}")
        print(f"Chunked: {'Yes' if len(review_chunks) > 1 else 'No'}")
        
        for i, chunk in enumerate(review_chunks):
            print(f"\n  Chunk {i + 1}:")
            print(f"    Token count: {chunk.token_count}")
            print(f"    Start index: {chunk.start_index}")
            print(f"    End index: {chunk.end_index}")
            print(f"    Text preview: {chunk.text[:100]}...")
    
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {len(df)} reviews processed into {total_chunks} chunks")
    print(f"Average chunks per review: {total_chunks / len(df):.2f}")
    print(f"{'=' * 80}")
    
    return True

if __name__ == "__main__":
    success = test_chonkie_chunking()
    if success:
        print("\n✅ Chonkie chunking test passed!")
    else:
        print("\n❌ Chonkie chunking test failed!")
