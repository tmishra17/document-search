#!/usr/bin/env python3
"""
Demonstration script showing the benefits of Chonkie chunking
"""
import pandas as pd
from chonkie import SentenceChunker
import re

def preprocess_review(text: str) -> str:
    """Removes unwanted HTML from the provided text"""
    return re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

def demonstrate_chunking():
    """Show before/after comparison of chunking"""
    
    chunker = SentenceChunker(
        chunk_size=512,
        chunk_overlap=50,
        min_sentences_per_chunk=2
    )
    
    # Load a longer review
    df = pd.read_csv("IMDB_Dataset.csv", nrows=1)
    review = df.iloc[0]['review']
    cleaned_review = preprocess_review(review)
    
    print("=" * 80)
    print("CHONKIE CHUNKING DEMONSTRATION")
    print("=" * 80)
    
    print("\n📄 ORIGINAL REVIEW (before chunking):")
    print("-" * 80)
    print(f"Length: {len(cleaned_review)} characters")
    print(f"Preview (first 500 chars):\n{cleaned_review[:500]}...")
    
    chunks = chunker.chunk(cleaned_review)
    
    print(f"\n\n✂️  AFTER CHUNKING:")
    print("-" * 80)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens")
    
    print("\n📝 CHUNK DETAILS:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  Chunk {i}:")
        print(f"    Tokens: {chunk.token_count}")
        print(f"    Characters: {len(chunk.text)}")
        print(f"    Range: {chunk.start_index} - {chunk.end_index}")
        print(f"    Text: {chunk.text[:200]}...")
    
    print("\n" + "=" * 80)
    print("💡 BENEFITS:")
    print("=" * 80)
    print("✓ Each chunk can be searched independently")
    print("✓ Better matching on specific review sections")
    print("✓ More precise semantic search results")
    print("✓ Maintains context with 50-token overlap")
    print("✓ Respects sentence boundaries for readability")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_chunking()
