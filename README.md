# Document Project

A simple movie review search system that gives users relevant results based on their query.

## How it works:

1. Load IMDB movie review dataset
2. Create embeddings using SentenceTransformers  
3. User enters search query
4. Find similar reviews using semantic search
5. Display matching results with similarity scores


## To run:

```bash
streamlit run DocumentSearchEngine.py
```
## To Download data set
run
```bash
kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
```
## Features:

- Semantic search (understands meaning, not just keywords)
- Similarity scoring 
- Clean HTML processing
- Adjustable similarity threshold
- Simple, easy-to-use interface
        # Movie Review Search Engine

A simple semantic search application for movie reviews using AI embeddings.

## How it Works

Uses AI to understand the meaning of your search, not just keywords. Searches through movie review text to find the most relevant matches.

## Features

- Semantic search (understands meaning, not just keywords)
- Similarity scoring 
- Clean HTML processing
- Adjustable similarity threshold
- Simple, easy-to-use interface

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run DocumentSearchEngine.py`
3. Upload your IMDB dataset when prompted
4. Start searching!

## Usage

- Enter your search query (e.g., "funny romantic comedy")  
- Adjust similarity threshold to control result relevance
- View matching reviews with similarity scores
```

### **5. Intelligent Search** 🎯
```python
# Convert user query to same embedding space
query_vector = matryoshka_model.encode(["funny alien movie"])

# Find most similar reviews in vector space
results = client.search(
    collection_name="movie_reviews", 
    query_vector=query_vector,
    limit=10,
    query_filter=sentiment_filter  # Optional: only positive/negative
)
```

## 🚀 **Quick Start Guide**

### **Prerequisites**
- Docker (for Qdrant database)
- Python with required packages

### **1. Start Qdrant Database**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### **2. Check System Status**
```bash
streamlit run qdrant_status.py
```

### **3. Run the Search Engine**
```bash
streamlit run DocumentSearchEngine.py
```

### **4. First Time Setup (Automatic)**
- App processes your IMDB dataset automatically
- Creates embeddings for all 50K reviews (takes 5-10 minutes)  
- Runs sentiment analysis on each review
- Stores everything in Qdrant for fast future searches

## 🎛️ **Advanced Features**

### **Search Controls:**
- **Similarity Threshold**: Higher = more relevant but fewer results
- **Sentiment Filter**: Show only positive/negative reviews  
- **Result Count**: 1-50 results per search
- **Real-time Filtering**: Instant results as you adjust settings

### **Rich Result Information:**
```
🎬 Review #1 - Similarity: 0.847
├── 📊 AI Sentiment: POSITIVE (0.92 confidence) 
├── 👤 Human Label: positive ✅ (AI got it right!)
├── 📝 Word Count: 247 words
└── 📄 Full review text with HTML cleaned
```

### **Performance Benefits:**
- ⚡ **Sub-second search** even with 50K+ reviews
- 🔄 **Persistent storage** - no reprocessing after restart  
- 💾 **Memory efficient** - vectors stored in database, not RAM
- 📈 **Scalable** - easily handle millions of reviews

## 🔬 **Technical Deep Dive**

### **Why Matryoshka Embeddings?**
- **Flexible dimensions**: Can use 128, 256, 512, or 1024 dimensions
- **Better performance**: Optimized for similarity search tasks
- **State-of-the-art**: mixedbread-ai model beats many alternatives

### **Why Qdrant Vector Database?**
- **Speed**: Optimized similarity search algorithms (HNSW indexing)
- **Filtering**: Combine vector search with metadata filters
- **Scalability**: Production-ready for millions of vectors  
- **Persistence**: Data survives crashes and restarts

### **Sentiment Analysis Pipeline:**
```python
# Uses HuggingFace transformer pipeline
classifier = pipeline("sentiment-analysis")

# For each review:
result = classifier(review_text[:512])  # Truncate for model limits
# Returns sentiment + confidence score
```

## 📊 **Example Queries That Work Well**

### **Genre/Theme Searches:**
- `"funny alien comedy"` → finds reviews about humorous sci-fi
- `"scary psychological thriller"` → finds horror with mental elements  
- `"romantic period drama"` → finds love stories in historical settings

### **Mood/Feeling Searches:**
- `"feel good uplifting"` → finds positive, inspiring reviews
- `"dark and depressing"` → finds reviews about serious, sad movies
- `"mindless fun entertainment"` → finds reviews about light, enjoyable films

### **Quality/Opinion Searches:**
- `"masterpiece brilliant acting"` → finds highly praising reviews
- `"boring waste of time"` → finds very negative reviews  
- `"overrated disappointment"` → finds reviews about overhyped movies

## 🔧 **Troubleshooting**

### **Qdrant Issues:**
```bash
# Check if Qdrant is running
curl localhost:6333

# Restart if needed  
docker restart <container_id>

# Check logs
docker logs <container_id>
```

### **Performance Tips:**
- Start with higher similarity threshold (0.4+) for most relevant results
- Use sentiment filtering to reduce search space
- Lower threshold (0.2-0.3) for broader, exploratory searches

### **Common Issues:**
- **First run slow**: Normal - processing 50K reviews takes time
- **No results**: Lower similarity threshold or try different keywords
- **Qdrant errors**: Make sure Docker container is running on port 6333


