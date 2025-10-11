#!/usr/bin/env python3
"""
Test script to verify Qdrant integration
"""
from qdrant_client import QdrantClient

def test_qdrant_connection():
    try:
        client = QdrantClient("localhost", port=6333)
        collections = client.get_collections()
        print("✅ Successfully connected to Qdrant!")
        print(f"Available collections: {[c.name for c in collections.collections]}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("Make sure Qdrant is running on localhost:6333")
        return False

if __name__ == "__main__":
    test_qdrant_connection()