"""
Pinecone Configuration Module
============================

This module handles all Pinecone vector database configurations including:
- Index creation and management
- Embedding configurations
- Namespace organization
- Vector operations setup
- Connection management

Usage:
    from config.pinecone_config import PineconeConfig, initialize_pinecone
    
    # Initialize Pinecone connection
    pc = initialize_pinecone()
    
    # Get index
    index = get_pinecone_index("medical-literature-index")
    
    # Use embedding config
    embed_config = EmbeddingConfig()
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec, PodSpec
    PINECONE_AVAILABLE = True
except ImportError:
    logger.warning("Pinecone library not available. Install with: pip install pinecone-client")
    PINECONE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Sentence transformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class IndexType(Enum):
    """Pinecone index types"""
    SERVERLESS = "serverless"
    POD = "pod"


class MetricType(Enum):
    """Vector similarity metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOTPRODUCT = "dotproduct"


class CloudProvider(Enum):
    """Cloud providers for Pinecone"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector database"""
    
    # API Configuration
    api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    environment: str = field(default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws"))
    
    # Index Configuration
    index_name: str = "medical-literature-index"
    dimension: int = 384  # Must match embedding model dimension
    metric: str = "cosine"
    index_type: str = "serverless"  # serverless or pod
    
    # Serverless Configuration
    cloud_provider: str = "aws"
    region: str = "us-east-1"
    
    # Pod Configuration (if using pod type)
    pod_type: str = "p1.x1"
    pods: int = 1
    replicas: int = 1
    shards: int = 1
    
    # Namespace Configuration
    namespaces: Dict[str, str] = field(default_factory=lambda: {
        "chest_xray": "chest-xray-cases",
        "skin_lesion": "skin-lesion-cases", 
        "brain_tumor": "brain-tumor-cases",
        "literature": "medical-literature",
        "case_studies": "clinical-cases",
        "embeddings": "text-embeddings"
    })
    
    # Performance Settings
    batch_size: int = 100  # For batch operations
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 60
    
    # Metadata Configuration
    metadata_config: Dict[str, Any] = field(default_factory=lambda: {
        "indexed": ["modality", "diagnosis", "confidence", "date", "source_type"],
        "stored": ["title", "abstract", "authors", "journal", "doi", "case_id"]
    })
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.api_key:
            logger.warning("Pinecone API key not provided. Set PINECONE_API_KEY environment variable.")
        
        # Validate dimension
        if self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        # Validate metric
        valid_metrics = ["cosine", "euclidean", "dotproduct"]
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Choose from: {valid_metrics}")
        
        # Validate index type
        valid_types = ["serverless", "pod"]
        if self.index_type not in valid_types:
            raise ValueError(f"Invalid index type. Choose from: {valid_types}")


@dataclass
class EmbeddingConfig:
    """Configuration for text embeddings"""
    
    # Model Configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_dimension: int = 384
    max_sequence_length: int = 512
    
    # Alternative medical domain models
    medical_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "biobert": {
            "name": "dmis-lab/biobert-base-cased-v1.2",
            "dimension": 768,
            "max_length": 512
        },
        "pubmedbert": {
            "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "dimension": 768,
            "max_length": 512
        },
        "scibert": {
            "name": "allenai/scibert_scivocab_uncased",
            "dimension": 768,
            "max_length": 512
        },
        "clinical_bert": {
            "name": "emilyalsentzer/Bio_ClinicalBERT",
            "dimension": 768,
            "max_length": 512
        }
    })
    
    # Processing Configuration
    batch_size: int = 32
    normalize_embeddings: bool = True
    use_medical_model: bool = True
    
    # Text Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    
    # Preprocessing
    remove_special_chars: bool = True
    lowercase: bool = True
    remove_stopwords: bool = False  # Keep medical terms
    
    def get_model_config(self, model_type: str = "default") -> Dict[str, Any]:
        """Get model configuration for specified type"""
        if model_type == "default":
            return {
                "name": self.model_name,
                "dimension": self.model_dimension,
                "max_length": self.max_sequence_length
            }
        elif model_type in self.medical_models:
            return self.medical_models[model_type]
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def initialize_pinecone(config: Optional[PineconeConfig] = None) -> Optional[Pinecone]:
    """
    Initialize Pinecone connection
    
    Args:
        config: Pinecone configuration (if None, uses default)
        
    Returns:
        Pinecone client instance or None if initialization fails
    """
    if not PINECONE_AVAILABLE:
        logger.error("Pinecone library not available")
        return None
    
    if config is None:
        config = PineconeConfig()
    
    if not config.api_key:
        logger.error("Pinecone API key not provided")
        return None
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=config.api_key)
        logger.info("Successfully initialized Pinecone connection")
        return pc
    
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        return None


def create_pinecone_index(
    pc: Pinecone,
    config: PineconeConfig,
    delete_if_exists: bool = False
) -> bool:
    """
    Create Pinecone index with specified configuration
    
    Args:
        pc: Pinecone client instance
        config: Pinecone configuration
        delete_if_exists: Whether to delete existing index
        
    Returns:
        True if index created successfully, False otherwise
    """
    try:
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_exists = any(idx.name == config.index_name for idx in existing_indexes)
        
        if index_exists:
            if delete_if_exists:
                logger.info(f"Deleting existing index: {config.index_name}")
                pc.delete_index(config.index_name)
                
                # Wait for deletion to complete
                while config.index_name in [idx.name for idx in pc.list_indexes()]:
                    time.sleep(1)
            else:
                logger.info(f"Index {config.index_name} already exists")
                return True
        
        # Create index specification
        if config.index_type == "serverless":
            spec = ServerlessSpec(
                cloud=config.cloud_provider,
                region=config.region
            )
        else:  # pod
            spec = PodSpec(
                environment=config.environment,
                pod_type=config.pod_type,
                pods=config.pods,
                replicas=config.replicas,
                shards=config.shards
            )
        
        # Create index
        logger.info(f"Creating Pinecone index: {config.index_name}")
        pc.create_index(
            name=config.index_name,
            dimension=config.dimension,
            metric=config.metric,
            spec=spec
        )
        
        # Wait for index to be ready
        while not pc.describe_index(config.index_name).status['ready']:
            time.sleep(1)
        
        logger.info(f"Successfully created index: {config.index_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create index: {str(e)}")
        return False


def get_pinecone_index(
    index_name: Optional[str] = None,
    config: Optional[PineconeConfig] = None
):
    """
    Get Pinecone index instance
    
    Args:
        index_name: Name of the index (if None, uses config default)
        config: Pinecone configuration
        
    Returns:
        Pinecone index instance or None
    """
    if config is None:
        config = PineconeConfig()
    
    if index_name is None:
        index_name = config.index_name
    
    pc = initialize_pinecone(config)
    if pc is None:
        return None
    
    try:
        index = pc.Index(index_name)
        logger.info(f"Successfully connected to index: {index_name}")
        return index
    except Exception as e:
        logger.error(f"Failed to connect to index {index_name}: {str(e)}")
        return None


class VectorOperations:
    """Utility class for vector operations with Pinecone"""
    
    def __init__(self, config: Optional[PineconeConfig] = None):
        self.config = config or PineconeConfig()
        self.index = get_pinecone_index(config=self.config)
        
    def upsert_vectors(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        namespace: str = "",
        batch_size: Optional[int] = None
    ) -> bool:
        """
        Upsert vectors to Pinecone index
        
        Args:
            vectors: List of (id, vector, metadata) tuples
            namespace: Namespace for the vectors
            batch_size: Batch size for upserts
            
        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.error("Index not available")
            return False
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        try:
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Format for Pinecone
                formatted_vectors = [
                    {
                        "id": vec_id,
                        "values": values,
                        "metadata": metadata
                    }
                    for vec_id, values, metadata in batch
                ]
                
                # Upsert batch
                self.index.upsert(
                    vectors=formatted_vectors,
                    namespace=namespace
                )
                
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}")
            return False
    
    def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query vectors from Pinecone index
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            namespace: Namespace to query
            filter_dict: Metadata filters
            include_metadata: Whether to include metadata
            include_values: Whether to include vector values
            
        Returns:
            List of query results
        """
        if self.index is None:
            logger.error("Index not available")
            return []
        
        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=include_values
            )
            
            return response.matches
            
        except Exception as e:
            logger.error(f"Failed to query vectors: {str(e)}")
            return []
    
    def delete_vectors(
        self,
        ids: List[str],
        namespace: str = ""
    ) -> bool:
        """
        Delete vectors from Pinecone index
        
        Args:
            ids: List of vector IDs to delete
            namespace: Namespace containing the vectors
            
        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.error("Index not available")
            return False
        
        try:
            self.index.delete(ids=ids, namespace=namespace)
            logger.info(f"Successfully deleted {len(ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics
        
        Returns:
            Index statistics dictionary
        """
        if self.index is None:
            logger.error("Index not available")
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {}


class EmbeddingGenerator:
    """Generate embeddings for text using various models"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Sentence transformers not available")
            return
        
        try:
            model_config = self.config.get_model_config(
                "biobert" if self.config.use_medical_model else "default"
            )
            
            self.model = SentenceTransformer(model_config["name"])
            logger.info(f"Loaded embedding model: {model_config['name']}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for list of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            logger.error("Embedding model not loaded")
            return np.array([])
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=len(texts) > 100
            )
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            return np.array([])
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.config.chunk_size:
                current_chunk += sentence + ". "
            else:
                if len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Add remaining chunk
        if len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks


# Configuration validation
def validate_pinecone_config(config: PineconeConfig) -> List[str]:
    """
    Validate Pinecone configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if not config.api_key:
        errors.append("API key is required")
    
    if not config.index_name:
        errors.append("Index name is required")
    
    if config.dimension <= 0:
        errors.append("Dimension must be positive")
    
    if config.metric not in ["cosine", "euclidean", "dotproduct"]:
        errors.append("Invalid metric type")
    
    if config.index_type not in ["serverless", "pod"]:
        errors.append("Invalid index type")
    
    return errors


# Export all classes and functions
__all__ = [
    'PineconeConfig',
    'EmbeddingConfig',
    'VectorOperations',
    'EmbeddingGenerator',
    'initialize_pinecone',
    'create_pinecone_index',
    'get_pinecone_index',
    'validate_pinecone_config',
    'IndexType',
    'MetricType',
    'CloudProvider'
]


# Example usage and testing functions
def test_pinecone_connection(config: Optional[PineconeConfig] = None) -> bool:
    """
    Test Pinecone connection and basic operations
    
    Args:
        config: Pinecone configuration
        
    Returns:
        True if all tests pass, False otherwise
    """
    if config is None:
        config = PineconeConfig()
    
    logger.info("Testing Pinecone connection...")
    
    # Test initialization
    pc = initialize_pinecone(config)
    if pc is None:
        logger.error("Failed to initialize Pinecone")
        return False
    
    # Test index operations
    try:
        # List indexes
        indexes = pc.list_indexes()
        logger.info(f"Found {len(indexes)} indexes")
        
        # Test index connection if exists
        if any(idx.name == config.index_name for idx in indexes):
            index = get_pinecone_index(config=config)
            if index is None:
                logger.error("Failed to connect to index")
                return False
            
            # Get stats
            vector_ops = VectorOperations(config)
            stats = vector_ops.get_index_stats()
            logger.info(f"Index stats: {stats}")
        
        logger.info("Pinecone connection test passed")
        return True
        
    except Exception as e:
        logger.error(f"Pinecone connection test failed: {str(e)}")
        return False


def create_sample_medical_embeddings(
    config: Optional[EmbeddingConfig] = None
) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    """
    Create sample medical text embeddings for testing
    
    Args:
        config: Embedding configuration
        
    Returns:
        List of (id, vector, metadata) tuples
    """
    if config is None:
        config = EmbeddingConfig()
    
    # Sample medical texts
    sample_texts = [
        "Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.",
        "Melanoma is a type of skin cancer that develops from the pigment-producing cells known as melanocytes.",
        "Glioblastoma is an aggressive type of cancer that can occur in the brain or spinal cord.",
        "Chest X-ray shows bilateral infiltrates consistent with pneumonia in a 65-year-old patient.",
        "Dermatoscopic examination reveals asymmetric pigmented lesion with irregular borders.",
        "MRI brain demonstrates enhancing mass lesion in the frontal lobe with surrounding edema."
    ]
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator(config)
    embeddings = embedding_gen.generate_embeddings(sample_texts)
    
    if len(embeddings) == 0:
        logger.error("Failed to generate sample embeddings")
        return []
    
    # Create vector tuples with metadata
    vectors = []
    for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
        metadata = {
            "text": text,
            "source_type": "sample",
            "modality": ["chest_xray", "skin_lesion", "brain_tumor"][i % 3],
            "created_at": "2024-01-01",
            "confidence": 0.95
        }
        
        vectors.append((f"sample_{i}", embedding.tolist(), metadata))
    
    logger.info(f"Created {len(vectors)} sample embeddings")
    return vectors


def setup_medical_rag_index(
    config: Optional[PineconeConfig] = None,
    create_sample_data: bool = True
) -> bool:
    """
    Complete setup of medical RAG index with sample data
    
    Args:
        config: Pinecone configuration
        create_sample_data: Whether to add sample data
        
    Returns:
        True if setup successful, False otherwise
    """
    if config is None:
        config = PineconeConfig()
    
    logger.info("Setting up medical RAG index...")
    
    # Initialize Pinecone
    pc = initialize_pinecone(config)
    if pc is None:
        return False
    
    # Create index
    if not create_pinecone_index(pc, config, delete_if_exists=True):
        return False
    
    # Add sample data if requested
    if create_sample_data:
        logger.info("Adding sample medical data...")
        
        # Generate sample embeddings
        sample_vectors = create_sample_medical_embeddings()
        if not sample_vectors:
            logger.warning("No sample data generated")
            return True
        
        # Upload to different namespaces
        vector_ops = VectorOperations(config)
        
        # Distribute samples across namespaces
        for i, (vec_id, vector, metadata) in enumerate(sample_vectors):
            namespace = list(config.namespaces.values())[i % len(config.namespaces)]
            
            success = vector_ops.upsert_vectors(
                [(vec_id, vector, metadata)],
                namespace=namespace
            )
            
            if not success:
                logger.warning(f"Failed to upload sample {vec_id}")
        
        logger.info("Sample data upload completed")
    
    logger.info("Medical RAG index setup completed successfully")
    return True


# Configuration examples for different use cases
class ConfigExamples:
    """Pre-configured examples for different deployment scenarios"""
    
    @staticmethod
    def development_config() -> PineconeConfig:
        """Configuration for development environment"""
        return PineconeConfig(
            index_name="medical-dev-index",
            dimension=384,
            metric="cosine",
            index_type="serverless",
            cloud_provider="aws",
            region="us-east-1"
        )
    
    @staticmethod
    def production_config() -> PineconeConfig:
        """Configuration for production environment"""
        return PineconeConfig(
            index_name="medical-prod-index",
            dimension=768,  # Larger dimension for better accuracy
            metric="cosine",
            index_type="pod",
            pod_type="p1.x2",  # Larger pod for production
            pods=2,
            replicas=2
        )
    
    @staticmethod
    def high_performance_config() -> PineconeConfig:
        """Configuration for high-performance requirements"""
        return PineconeConfig(
            index_name="medical-hp-index",
            dimension=1024,
            metric="cosine",
            index_type="pod",
            pod_type="p2.x1",
            pods=4,
            replicas=3,
            batch_size=200
        )
    
    @staticmethod
    def biobert_embedding_config() -> EmbeddingConfig:
        """Configuration using BioBERT for medical domain"""
        return EmbeddingConfig(
            model_name="dmis-lab/biobert-base-cased-v1.2",
            model_dimension=768,
            max_sequence_length=512,
            use_medical_model=True,
            chunk_size=256,
            normalize_embeddings=True
        )
    
    @staticmethod
    def clinical_bert_config() -> EmbeddingConfig:
        """Configuration using Clinical BERT"""
        return EmbeddingConfig(
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            model_dimension=768,
            max_sequence_length=512,
            use_medical_model=True,
            chunk_size=512,
            normalize_embeddings=True
        )


# Utility functions for common operations
def migrate_index_data(
    source_config: PineconeConfig,
    target_config: PineconeConfig,
    batch_size: int = 100
) -> bool:
    """
    Migrate data from one Pinecone index to another
    
    Args:
        source_config: Source index configuration
        target_config: Target index configuration
        batch_size: Batch size for migration
        
    Returns:
        True if migration successful, False otherwise
    """
    logger.info(f"Migrating data from {source_config.index_name} to {target_config.index_name}")
    
    # Get source and target operations
    source_ops = VectorOperations(source_config)
    target_ops = VectorOperations(target_config)
    
    if source_ops.index is None or target_ops.index is None:
        logger.error("Failed to connect to source or target index")
        return False
    
    try:
        # Get source index stats
        source_stats = source_ops.get_index_stats()
        total_vectors = source_stats.get('total_vector_count', 0)
        
        if total_vectors == 0:
            logger.info("No vectors to migrate")
            return True
        
        logger.info(f"Migrating {total_vectors} vectors...")
        
        # Note: This is a simplified migration
        # In practice, you'd need to iterate through all vectors
        # Pinecone doesn't provide a direct way to export all vectors
        # You'd typically maintain your own backup or use Pinecone's backup features
        
        logger.warning("Complete migration requires custom implementation based on your data source")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        return False


def cleanup_old_vectors(
    config: PineconeConfig,
    days_old: int = 30,
    namespace: str = ""
) -> bool:
    """
    Clean up old vectors based on metadata timestamp
    
    Args:
        config: Pinecone configuration
        days_old: Delete vectors older than this many days
        namespace: Namespace to clean up
        
    Returns:
        True if cleanup successful, False otherwise
    """
    logger.info(f"Cleaning up vectors older than {days_old} days in namespace '{namespace}'")
    
    vector_ops = VectorOperations(config)
    if vector_ops.index is None:
        logger.error("Failed to connect to index")
        return False
    
    try:
        # This is a placeholder - actual implementation would require
        # querying vectors with date filters and deleting them
        logger.warning("Cleanup function requires implementation based on your metadata structure")
        return True
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return False


# Configuration validation with detailed error messages
def comprehensive_config_validation(
    pinecone_config: PineconeConfig,
    embedding_config: EmbeddingConfig
) -> Dict[str, List[str]]:
    """
    Comprehensive validation of all configurations
    
    Args:
        pinecone_config: Pinecone configuration
        embedding_config: Embedding configuration
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "pinecone_errors": validate_pinecone_config(pinecone_config),
        "embedding_errors": [],
        "compatibility_errors": []
    }
    
    # Validate embedding config
    if embedding_config.model_dimension <= 0:
        results["embedding_errors"].append("Model dimension must be positive")
    
    if embedding_config.chunk_size <= 0:
        results["embedding_errors"].append("Chunk size must be positive")
    
    if embedding_config.batch_size <= 0:
        results["embedding_errors"].append("Batch size must be positive")
    
    # Check compatibility
    if pinecone_config.dimension != embedding_config.model_dimension:
        results["compatibility_errors"].append(
            f"Pinecone dimension ({pinecone_config.dimension}) doesn't match "
            f"embedding dimension ({embedding_config.model_dimension})"
        )
    
    return results


# Export updated __all__
__all__ = [
    'PineconeConfig',
    'EmbeddingConfig',
    'VectorOperations',
    'EmbeddingGenerator',
    'initialize_pinecone',
    'create_pinecone_index',
    'get_pinecone_index',
    'validate_pinecone_config',
    'test_pinecone_connection',
    'create_sample_medical_embeddings',
    'setup_medical_rag_index',
    'ConfigExamples',
    'migrate_index_data',
    'cleanup_old_vectors',
    'comprehensive_config_validation',
    'IndexType',
    'MetricType',
    'CloudProvider'
]