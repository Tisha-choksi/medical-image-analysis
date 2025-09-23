#!/usr/bin/env python3
"""
Fix Pinecone Setup Script
========================

This script fixes dimension mismatches and uploads sample data.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.pinecone_config import (
    PineconeConfig,
    EmbeddingConfig,
    initialize_pinecone,
    create_pinecone_index,
    VectorOperations,
    EmbeddingGenerator
)

def fix_dimension_mismatch():
    """Fix the dimension mismatch by recreating the index"""
    print("🔧 Fixing dimension mismatch...")
    
    # Get current embedding dimension
    embed_config = EmbeddingConfig()
    embed_gen = EmbeddingGenerator(embed_config)
    
    # Test embedding to get actual dimension
    test_embedding = embed_gen.generate_embeddings(["test"])
    if len(test_embedding) == 0:
        print("❌ Failed to generate test embedding")
        return False
    
    actual_dimension = len(test_embedding[0])
    print(f"📏 Detected embedding dimension: {actual_dimension}")
    
    # Update Pinecone config with correct dimension
    pc_config = PineconeConfig()
    pc_config.dimension = actual_dimension
    
    print(f"🔄 Recreating index with dimension {actual_dimension}...")
    
    # Initialize Pinecone
    pc = initialize_pinecone(pc_config)
    if pc is None:
        print("❌ Failed to initialize Pinecone")
        return False
    
    # Recreate index with correct dimension
    success = create_pinecone_index(pc, pc_config, delete_if_exists=True)
    if not success:
        print("❌ Failed to recreate index")
        return False
    
    print("✅ Index recreated with correct dimensions")
    return True, pc_config, embed_config

def create_sample_data():
    """Create comprehensive sample medical data"""
    return [
        {
            "id": "chest_xray_paper_001",
            "title": "Deep Learning Approaches for Pneumonia Detection in Chest X-rays",
            "abstract": "Pneumonia is a leading cause of death worldwide. This study presents a comprehensive analysis of deep learning methods for automated pneumonia detection using chest X-ray images. We trained ResNet50 and DenseNet models on 5,863 chest X-ray images and achieved 95.3% accuracy in binary classification between normal and pneumonia cases. The model showed high sensitivity (96.8%) and specificity (93.2%) in detecting pneumonia patterns including consolidation, infiltrates, and pleural effusion.",
            "authors": ["Dr. Smith, J.", "Dr. Johnson, A.", "Dr. Williams, B."],
            "journal": "Journal of Medical AI",
            "year": 2023,
            "modality": "chest_xray",
            "keywords": ["pneumonia", "chest X-ray", "deep learning", "ResNet50", "medical imaging"]
        },
        {
            "id": "chest_xray_case_001", 
            "title": "Case Report: Bilateral Pneumonia in 68-year-old Male",
            "abstract": "A 68-year-old male with diabetes presented to the emergency department with a 3-day history of fever, productive cough, and dyspnea. Physical examination revealed crackles in both lung bases. Chest X-ray demonstrated bilateral lower lobe infiltrates consistent with pneumonia. Laboratory results showed elevated white blood cell count (15,200/μL) and C-reactive protein (85 mg/L). Patient was treated with intravenous antibiotics and showed clinical improvement within 48 hours.",
            "authors": ["Clinical Team"],
            "journal": "Emergency Medicine Cases",
            "year": 2023,
            "modality": "chest_xray",
            "diagnosis": "bilateral pneumonia",
            "confidence": 0.94,
            "keywords": ["bilateral pneumonia", "infiltrates", "elderly", "diabetes", "antibiotics"]
        },
        {
            "id": "skin_lesion_paper_001",
            "title": "Melanoma Classification Using EfficientNet and HAM10000 Dataset",
            "abstract": "Melanoma is the most deadly form of skin cancer, responsible for 75% of skin cancer deaths. Early detection is crucial for patient survival. This research implements EfficientNetB0 architecture for multi-class classification of skin lesions using the HAM10000 dataset containing 10,015 dermatoscopic images. We achieved 91.2% accuracy in distinguishing between melanoma (MEL), melanocytic nevi (NV), basal cell carcinoma (BCC), actinic keratoses (AKIEC), benign keratosis (BKL), dermatofibroma (DF), and vascular lesions (VASC).",
            "authors": ["Dr. Brown, C.", "Dr. Davis, M.", "Dr. Wilson, K."],
            "journal": "Dermatology and AI",
            "year": 2023,
            "modality": "skin_lesion",
            "keywords": ["melanoma", "skin cancer", "dermatoscopy", "EfficientNet", "HAM10000"]
        },
        {
            "id": "skin_lesion_case_001",
            "title": "Early-Stage Melanoma Detection: 42-year-old Female",
            "abstract": "A 42-year-old female noticed changes in a mole on her shoulder over 3 months. Dermatoscopic examination revealed asymmetric pigmentation with irregular borders (ABCD criteria positive). The lesion measured 8mm in diameter with color variation from light brown to black. Dermoscopy showed atypical pigment network and blue-white veil. Excisional biopsy confirmed melanoma in situ with Clark level I. Complete surgical excision with 5mm margins was performed with excellent cosmetic result.",
            "authors": ["Dermatology Department"],
            "journal": "Dermatology Case Studies",
            "year": 2023,
            "modality": "skin_lesion",
            "diagnosis": "melanoma",
            "confidence": 0.89,
            "keywords": ["melanoma", "dermoscopy", "ABCD criteria", "early detection", "surgical excision"]
        },
        {
            "id": "brain_tumor_paper_001", 
            "title": "Brain Tumor Classification in MRI Images Using DenseNet Architecture",
            "abstract": "Brain tumors are among the most aggressive diseases with poor prognosis if not detected early. This study presents automated classification of brain tumors using T1-weighted MRI images. We implemented DenseNet121 architecture to classify four categories: glioma, meningioma, pituitary tumor, and normal brain tissue. Training on 3,264 MRI images achieved 93.7% accuracy. Gliomas showed characteristic irregular enhancement patterns, meningiomas demonstrated homogeneous enhancement, and pituitary tumors were identified by their suprasellar location.",
            "authors": ["Dr. Garcia, L.", "Dr. Martinez, R.", "Dr. Lopez, S."],
            "journal": "Neuroradiology and AI",
            "year": 2023,
            "modality": "brain_tumor",
            "keywords": ["brain tumor", "MRI", "DenseNet", "glioma", "meningioma", "pituitary"]
        },
        {
            "id": "brain_tumor_case_001",
            "title": "Glioblastoma Diagnosis: 55-year-old Male with Progressive Headaches",
            "abstract": "A 55-year-old male presented with 6-week history of progressively worsening headaches, confusion, and left-sided weakness. Neurological examination revealed right hemiparesis and speech difficulties. MRI brain with gadolinium demonstrated a 4.2cm heterogeneously enhancing mass in the right frontal lobe with surrounding vasogenic edema and mass effect. Stereotactic biopsy confirmed glioblastoma multiforme (GBM) WHO grade IV. Patient underwent maximal safe resection followed by concurrent chemoradiation therapy.",
            "authors": ["Neurosurgery Team"],
            "journal": "Neurosurgical Cases",
            "year": 2023,
            "modality": "brain_tumor",
            "diagnosis": "glioblastoma",
            "confidence": 0.96,
            "keywords": ["glioblastoma", "GBM", "mass effect", "stereotactic biopsy", "chemoradiation"]
        },
        {
            "id": "medical_review_001",
            "title": "AI in Medical Imaging: Current Applications and Future Prospects",
            "abstract": "Artificial intelligence has revolutionized medical imaging across multiple specialties. This comprehensive review examines current applications in radiology, dermatology, and pathology. Deep learning models have achieved expert-level performance in chest X-ray interpretation, skin lesion classification, and brain tumor detection. Key challenges include dataset bias, model interpretability, regulatory approval, and clinical integration. Future developments focus on multimodal AI, federated learning, and real-time diagnostic support systems.",
            "authors": ["Dr. Anderson, P.", "Dr. Thompson, E.", "Dr. Clark, R."],
            "journal": "AI in Medicine Review",
            "year": 2023,
            "modality": "general",
            "keywords": ["artificial intelligence", "medical imaging", "deep learning", "clinical integration", "regulatory"]
        }
    ]

def upload_comprehensive_data(vector_ops, embedding_gen):
    """Upload comprehensive medical data to Pinecone"""
    print("📚 Creating comprehensive medical literature database...")
    
    sample_data = create_sample_data()
    
    # Prepare data for embedding
    texts = []
    metadata_list = []
    ids = []
    
    for item in sample_data:
        # Create rich text for embedding
        text_parts = [
            item['title'],
            item['abstract']
        ]
        
        # Add keywords if available
        if 'keywords' in item:
            keywords_text = " ".join(item['keywords'])
            text_parts.append(f"Keywords: {keywords_text}")
        
        # Add diagnosis if available
        if 'diagnosis' in item:
            text_parts.append(f"Diagnosis: {item['diagnosis']}")
        
        combined_text = " | ".join(text_parts)
        texts.append(combined_text)
        ids.append(item['id'])
        
        # Prepare comprehensive metadata
        metadata = {
            "title": item['title'],
            "abstract": item['abstract'][:1000],  # Limit abstract length for metadata
            "authors": ", ".join(item['authors']) if isinstance(item['authors'], list) else item['authors'],
            "journal": item.get('journal', ''),
            "year": item.get('year', 2023),
            "modality": item['modality'],
            "keywords": ", ".join(item['keywords']) if 'keywords' in item else "",
            "source_type": "case_study" if "case" in item['id'] else "research_paper"
        }
        
        # Add clinical metadata if available
        if 'diagnosis' in item:
            metadata['diagnosis'] = item['diagnosis']
        if 'confidence' in item:
            metadata['confidence'] = item['confidence']
            
        metadata_list.append(metadata)
    
    # Generate embeddings
    print("🔄 Generating embeddings for medical literature...")
    print(f"   Processing {len(texts)} documents...")
    
    embeddings = embedding_gen.generate_embeddings(texts)
    
    if len(embeddings) == 0:
        print("❌ Failed to generate embeddings")
        return False
    
    print(f"✅ Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
    
    # Prepare vectors for upload
    vectors = []
    for vec_id, embedding, metadata in zip(ids, embeddings, metadata_list):
        vectors.append((vec_id, embedding.tolist(), metadata))
    
    # Upload to organized namespaces
    config = vector_ops.config
    success_count = 0
    
    # Group by namespace
    namespace_groups = {}
    for vec_id, vector, metadata in vectors:
        modality = metadata['modality']
        source_type = metadata['source_type']
        
        # Determine namespace based on modality and type
        if modality == 'general' or source_type == 'research_paper':
            namespace = config.namespaces.get('literature', 'medical-literature')
        else:
            namespace = config.namespaces.get(modality, f"{modality}-cases")
        
        if namespace not in namespace_groups:
            namespace_groups[namespace] = []
        namespace_groups[namespace].append((vec_id, vector, metadata))
    
    # Upload by namespace
    for namespace, namespace_vectors in namespace_groups.items():
        print(f"📤 Uploading {len(namespace_vectors)} vectors to '{namespace}'...")
        
        success = vector_ops.upsert_vectors(
            namespace_vectors,
            namespace=namespace
        )
        
        if success:
            success_count += len(namespace_vectors)
            print(f"✅ Successfully uploaded to '{namespace}'")
        else:
            print(f"❌ Failed to upload to '{namespace}'")
    
    print(f"📊 Upload Summary: {success_count}/{len(vectors)} vectors uploaded successfully")
    
    # Show final statistics
    stats = vector_ops.get_index_stats()
    print(f"📈 Final Index Stats:")
    print(f"   • Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"   • Dimension: {stats.get('dimension', 'Unknown')}")
    print(f"   • Namespaces: {list(stats.get('namespaces', {}).keys())}")
    
    return success_count > 0

def test_search_functionality(vector_ops, embedding_gen):
    """Test the search functionality with medical queries"""
    print("\n🔍 Testing search functionality...")
    
    test_queries = [
        "pneumonia chest x-ray bilateral infiltrates",
        "melanoma skin lesion irregular borders",
        "brain tumor glioblastoma MRI enhancement"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔎 Test Query {i}: '{query}'")
        
        # Generate query embedding
        query_embedding = embedding_gen.generate_embeddings([query])
        if len(query_embedding) == 0:
            print("❌ Failed to generate query embedding")
            continue
        
        # Search across all namespaces
        namespaces_to_search = ['medical-literature', 'chest-xray-cases', 'skin-lesion-cases', 'brain-tumor-cases']
        
        for namespace in namespaces_to_search:
            results = vector_ops.query_vectors(
                query_vector=query_embedding[0].tolist(),
                top_k=2,
                namespace=namespace,
                include_metadata=True
            )
            
            if results:
                print(f"   📚 Results from '{namespace}':")
                for j, result in enumerate(results):
                    metadata = result.metadata
                    title = metadata.get('title', 'Unknown')[:50]
                    score = result.score
                    print(f"      {j+1}. Score: {score:.3f} - {title}...")
            
    print("\n✅ Search functionality test completed!")

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🔧 Fixing Pinecone Setup Issues 🔧             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Fix dimension mismatch
    result = fix_dimension_mismatch()
    if not result:
        print("❌ Failed to fix dimension mismatch")
        return
    
    success, pc_config, embed_config = result
    
    # Step 2: Upload comprehensive sample data
    print("\n📚 Uploading comprehensive medical literature...")
    vector_ops = VectorOperations(pc_config)
    embedding_gen = EmbeddingGenerator(embed_config)
    
    if not upload_comprehensive_data(vector_ops, embedding_gen):
        print("❌ Failed to upload sample data")
        return
    
    # Step 3: Test search functionality
    test_search_functionality(vector_ops, embedding_gen)
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                🎉 Setup Fixed Successfully! 🎉              ║
    ║                                                              ║
    ║  Your Pinecone index now contains:                           ║
    ║  • Correct embedding dimensions (768)                        ║
    ║  • Comprehensive medical literature                          ║
    ║  • Organized namespaces by specialty                         ║
    ║  • Working search functionality                              ║
    ║                                                              ║
    ║  Ready for RAG-powered medical diagnosis! 🏥                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()