# First import the chunker you want from Chonkie
from chonkie import RecursiveChunker, SentenceChunker

# Initialize the chunker
chunker = SentenceChunker()

# Chunk some text
chunks = chunker("Chonkie is the goodest boi! My favorite chunking hippo hehe.")
print(chunks)

# Access chunks
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")
    print(chunk)