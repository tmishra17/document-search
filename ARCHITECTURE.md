# Chonkie Implementation - Visual Architecture

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          IMDB DATASET (50K Reviews)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────┐
                          │  Load in batches │
                          │   (BATCH_SIZE)   │
                          └──────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────────┐
                          │  HTML Preprocessing      │
                          │  regex: <br\s*/?>  → \n  │
                          └──────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────────┐
                    │     CHONKIE SENTENCE CHUNKER      │
                    │                                    │
                    │  • chunk_size: 512 tokens          │
                    │  • chunk_overlap: 50 tokens        │
                    │  • min_sentences: 2                │
                    │  • respects sentence boundaries    │
                    └────────────────────────────────────┘
                                     │
                    ┌────────────────┴─────────────────┐
                    ▼                                  ▼
         ┌─────────────────┐              ┌─────────────────┐
         │  Short Review   │              │   Long Review   │
         │   (< 512 tok)   │              │   (> 512 tok)   │
         └─────────────────┘              └─────────────────┘
                    │                                  │
                    ▼                                  ▼
           ┌─────────────┐                    ┌───────────────┐
           │  1 Chunk    │                    │  2-4 Chunks   │
           └─────────────┘                    └───────────────┘
                    │                                  │
                    └────────────────┬─────────────────┘
                                     ▼
                          ┌──────────────────────┐
                          │  Sentiment Analysis  │
                          │  (once per review)   │
                          └──────────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────────┐
                          │  Create Chunk Metadata     │
                          │                            │
                          │  • review (chunk text)     │
                          │  • original_review         │
                          │  • sentiment               │
                          │  • sentiment_score         │
                          │  • chunk_index             │
                          │  • is_chunked              │
                          └────────────────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────────┐
                          │  Sentence Transformer      │
                          │  (all-MiniLM-L6-v2)        │
                          │  Create embeddings         │
                          └────────────────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────────┐
                          │  Store as PyTorch Tensor   │
                          │  + DataFrame               │
                          └────────────────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────────┐
                          │  Save to pickle file       │
                          │  (review.pkl)              │
                          └────────────────────────────┘
```

## Search Flow

```
┌──────────────────┐
│  User Query      │
│  "thriller movie"│
└──────────────────┘
         │
         ▼
┌────────────────────┐
│  Query Embedding   │
│  (sentence-trans)  │
└────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Semantic Search            │
│  (cosine similarity)        │
│  against ALL chunk vectors  │
└─────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Get Top-K Results           │
│  (includes chunks from       │
│   different reviews)         │
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Filter by Threshold         │
│  Apply Sentiment Boost       │
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Return Matching Chunks      │
│  (not full reviews)          │
└──────────────────────────────┘
```

## Example: Long Review Processing

```
INPUT REVIEW (1731 chars):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"One of the other reviewers has mentioned that after watching just 1 Oz 
episode you'll be hooked. They are right, as this is exactly what happened 
with me.<br /><br />The first thing that struck me about Oz was its 
brutality and unflinching scenes of violence, which set in right from the 
word GO. Trust me, this is not a show for the faint hearted or timid. This 
show pulls no punches with regards to drugs, sex or violence. Its is 
hardcore, in the classic use of the word.<br /><br />It is called OZ as 
that is the nickname given to the Oswald Maximum Security State Penitentary. 
It focuses mainly on Emerald City..."

                                    ↓
                          [HTML Preprocessing]
                                    ↓
                          [Chonkie Chunking]
                                    ↓

OUTPUT (4 CHUNKS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Chunk 1 (425 tokens):
├─ "One of the other reviewers has mentioned that after watching just 1 Oz..."
├─ sentiment: "positive"
├─ chunk_index: 0
└─ is_chunked: true

Chunk 2 (323 tokens):
├─ "Its is hardcore, in the classic use of the word. It is called OZ..."
├─ sentiment: "positive"
├─ chunk_index: 425
└─ is_chunked: true

Chunk 3 (412 tokens):
├─ "Em City is home to many..Aryans, Muslims, gangstas, Latinos..."
├─ sentiment: "positive"
├─ chunk_index: 748
└─ is_chunked: true

Chunk 4 (601 tokens):
├─ "The first episode I ever saw struck me as so nasty it was surreal..."
├─ sentiment: "positive"
├─ chunk_index: 1160
└─ is_chunked: true

Each chunk now gets its own embedding vector for independent search!
```

## Key Benefits Illustrated

```
🔍 SEARCH QUALITY IMPROVEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WITHOUT Chonkie:
  Query: "brutal prison violence"
  ❌ Matches entire review (1731 chars)
  ❌ Average relevance due to mixed content
  ❌ Cannot pinpoint specific mentions

WITH Chonkie:
  Query: "brutal prison violence"
  ✅ Matches Chunk 1 specifically (425 tokens)
  ✅ High relevance for that section
  ✅ Precise match on violent scene description
  ✅ Better ranking and user experience


📊 EMBEDDING QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WITHOUT Chonkie:
  1 Review → 1 Dense Vector (represents entire review)
  ❌ Mixed signals in embedding
  ❌ Long reviews dilute specific content

WITH Chonkie:
  1 Review → 2-4 Focused Vectors (each represents a section)
  ✅ Each vector captures specific topic
  ✅ Better semantic representation
  ✅ More nuanced search results


🔄 CONTEXT PRESERVATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

50-token overlap ensures:
  Chunk 1: [tokens 0-512]
  Chunk 2: [tokens 462-974]     ← 50 tokens overlap
  Chunk 3: [tokens 924-1436]    ← 50 tokens overlap

  ✅ No context lost between chunks
  ✅ Maintains semantic continuity
  ✅ Better understanding of transitions
```
