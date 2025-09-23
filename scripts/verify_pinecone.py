#!/usr/bin/env python3
"""
Pinecone Verification Script
===========================

Quick script to verify your Pinecone setup is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.pinecone_config import (
    PineconeConfig,
    EmbeddingConfig,
    VectorOperations,
    EmbeddingGenerator
)

def main():
    print("🔍 Verifying Pinecone setup...")
    
    # Initialize configurations
    pc_config = PineconeConfig()
    embed_config = EmbeddingConfig()
    
    # Test vector operations
    vector_ops = VectorOperations(pc_config)
    
    if vector_ops.index is None:
        print("❌ Failed to connect to Pinecone index")
        return False
    
    # Get index stats
    stats = vector_ops.get_index_stats()
    print(f"📊 Index Stats: {stats}")
    
    # Test embedding generation
    embedding_gen = EmbeddingGenerator(embed_config)
    test_text = "Patient presents with chest pain and shortness of breath."
    
    embeddings = embedding_gen.generate_embeddings([test_text])
    
    if len(embeddings) > 0:
        print(f"✅ Embedding generation working (dimension: {len(embeddings[0])})")
        
        # Test similarity search
        results = vector_ops.query_vectors(
            query_vector=embeddings[0].tolist(),
            top_k=3,
            namespace="medical-literature"
        )
        
        print(f"🔍 Found {len(results)} similar documents")
        for i, result in enumerate(results):
            print(f"   {i+1}. Score: {result.score:.3f} - ID: {result.id}")
    else:
        print("❌ Embedding generation failed")
        return False
    
    print("✅ Pinecone setup verification complete!")
    return True

if __name__ == "__main__":
    main()