# 😃 Awesome Movie Finder

A semantic search application for movie reviews using AI embeddings and sentiment analysis.

## How it Works

Uses AI to understand the meaning of your search, not just keywords. Combines semantic similarity with intelligent sentiment matching to find the most relevant reviews.

**Enhanced with Chonkie Chunking**: Long reviews are intelligently split into semantic chunks using the Chonkie library, enabling more precise search results by matching specific sections of reviews rather than entire reviews.

## Features

- **Semantic search** - understands meaning, not just keywords
- **Intelligent review chunking** - uses Chonkie to split long reviews into semantic chunks for precise matching
- **Intelligent sentiment analysis** - matches query emotion with review sentiment
- **Dynamic scoring** - combines similarity + sentiment confidence
- **Clean HTML processing** - removes unwanted tags and formatting
- **Adjustable thresholds** - fine-tune result relevance
- **Real-time results** - instant search across 50K+ reviews

## Quick Start

### Prerequisites
```bash
# Download dataset
kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Install dependencies  
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run DocumentSearchEngine.py
```

### First Time Setup
- App automatically processes IMDB dataset (5-10 minutes)
- Uses Chonkie to intelligently chunk long reviews into semantic segments
- Creates embeddings for all review chunks
- Runs sentiment analysis with confidence scoring
- Caches everything for fast future searches

## Advanced Scoring System

### Intelligent Review Chunking with Chonkie
- **Semantic Chunking** - Long reviews are split into coherent, meaningful segments
- **Context Preservation** - 50-token overlap maintains context between chunks
- **Optimal Chunk Size** - 512 tokens per chunk for efficient embedding
- **Sentence Boundaries** - Chunks respect sentence boundaries for readability
- **Better Matching** - Search matches specific sections rather than entire reviews

### How Sentiment Enhances Search
1. **Query Analysis** - AI determines if you want positive/negative reviews
2. **Confidence Weighting** - Higher confidence = bigger impact on ranking
3. **Smart Boosting** - Reviews matching your sentiment get prioritized
4. **Balanced Results** - Semantic relevance remains primary factor

### Example Queries
- `"amazing thriller movie"` → boosts positive thriller reviews
- `"terrible horror film"` → prioritizes negative horror reviews  
- `"okay romantic comedy"` → neutral sentiment, minimal boost

## Troubleshooting

- **First run slow**: Normal - processing takes time
- **No results**: Lower similarity threshold (try 0.2-0.3)
- **Memory issues**: App uses CPU mode automatically

## Contributing
Feel free to submit issues and enhancement requests!
