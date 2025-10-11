import streamlit as st
from qdrant_client import QdrantClient

def main():
    st.title("🔧 Qdrant Vector Database Status")
    
    try:
        client = QdrantClient("localhost", port=6333)
        collections = client.get_collections()
        
        st.success("✅ Successfully connected to Qdrant!")
        
        if collections.collections:
            st.subheader("Available Collections:")
            for collection in collections.collections:
                col_info = client.get_collection(collection.name)
                st.write(f"**{collection.name}**")
                st.write(f"- Points: {col_info.points_count:,}")
                st.write(f"- Vector size: {col_info.config.params.vectors.size}")
                st.write(f"- Distance: {col_info.config.params.vectors.distance}")
                st.write("---")
        else:
            st.info("No collections found. Run the main app to create the movie_reviews collection.")
    
    except Exception as e:
        st.error(f"❌ Cannot connect to Qdrant: {e}")
        st.markdown("""
        **To start Qdrant:**
        ```bash
        docker run -p 6333:6333 qdrant/qdrant
        ```
        """)

if __name__ == "__main__":
    main()