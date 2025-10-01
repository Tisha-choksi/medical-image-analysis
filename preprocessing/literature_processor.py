"""
Medical Literature Processing Module - Fixed Version
==================================================

This module provides text processing capabilities for medical literature including:
- Text cleaning and preprocessing
- Medical entity extraction
- Literature chunking for RAG pipeline
- Citation parsing and metadata extraction

Classes:
- LiteratureProcessor: Main class for processing medical literature
- TextCleaner: Text cleaning utilities
- MedicalTextProcessor: Medical domain-specific text processing
- CitationExtractor: Extract and parse medical citations

Usage:
    from preprocessing.literature_processor import LiteratureProcessor
    
    processor = LiteratureProcessor()
    processed_text = processor.process_file('medical_paper.pdf')
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

# Safe imports with availability checks
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("NLTK not available. Some functionality will be limited.")

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    # Don't import specific models at module level
    import scispacy
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import PDF extractor at module level to avoid circular imports
try:
    from .pdf_extractor import PDFExtractor, PDFMetadata
    PDF_EXTRACTOR_AVAILABLE = True
except ImportError:
    PDF_EXTRACTOR_AVAILABLE = False
    PDFMetadata = None

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Container for processed document data"""
    title: str
    abstract: str
    full_text: str
    chunks: List[str]
    metadata: Dict[str, Any]
    entities: Dict[str, List[str]]
    citations: List[Dict[str, Any]]
    keywords: List[str]
    processed_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDocument':
        """Create from dictionary"""
        return cls(**data)


class TextCleaner:
    """Text cleaning utilities for medical literature"""
    
    def __init__(self, 
                 remove_special_chars: bool = True,
                 normalize_whitespace: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 preserve_medical_terms: bool = True):
        """
        Initialize text cleaner
        
        Args:
            remove_special_chars: Remove special characters
            normalize_whitespace: Normalize whitespace
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            preserve_medical_terms: Preserve medical terminology
        """
        self.remove_special_chars = remove_special_chars
        self.normalize_whitespace = normalize_whitespace
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.preserve_medical_terms = preserve_medical_terms
        
        # Medical term patterns to preserve
        self.medical_patterns = [
            r'\b\d+\.?\d*\s*(?:mg|ml|g|μg|mcg|IU|units?)\b',  # Dosages
            r'\bp\s*[<>=]\s*0\.\d+\b',  # p-values
            r'\bOR\s*=\s*\d+\.?\d*\b',  # Odds ratios
            r'\b(?:CT|MRI|X-ray|PET|SPECT)\b',  # Imaging
            r'\b(?:ICD|CPT|DRG)-\d+\b',  # Medical codes
            r'\b\d+\.?\d*\s*°[CF]\b',  # Temperatures
            r'\b\d+/\d+\s*mmHg\b',  # Blood pressure
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean text while preserving medical terminology
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Store medical terms temporarily with unique placeholders
        preserved_terms = {}
        if self.preserve_medical_terms:
            for i, pattern in enumerate(self.medical_patterns):
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for j, match in enumerate(matches):
                        placeholder = f"__MEDICAL_TERM_{i}_{j}_{hash(match) % 1000}__"
                        preserved_terms[placeholder] = match
                        # Use re.sub with count=1 to avoid multiple replacements
                        text = re.sub(re.escape(match), placeholder, text, count=1)
                except re.error as e:
                    logger.warning(f"Regex error in medical pattern {i}: {e}")
                    continue
        
        # Remove URLs
        if self.remove_urls:
            url_patterns = [
                r'https?://[^\s<>"]{2,}',
                r'www\.[^\s<>"]{2,}',
                r'[^\s<>"]+\.(?:com|org|edu|gov|net|mil)[^\s<>"]*'
            ]
            for pattern in url_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters but preserve sentence structure
        if self.remove_special_chars:
            # Keep essential punctuation and parentheses
            text = re.sub(r'[^\w\s.,;:!?()\-/]', ' ', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Restore medical terms
        for placeholder, original_term in preserved_terms.items():
            text = text.replace(placeholder, original_term)
        
        return text
    
    def remove_references(self, text: str) -> str:
        """Remove reference citations from text"""
        if not text:
            return ""
        
        # Remove numbered references like [1], [2-4], (Smith et al., 2020)
        patterns = [
            r'\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\]',
            r'\(\s*[A-Z][a-z]+\s+et\s+al\.?,?\s*\d{4}[a-z]?\s*\)',
            r'\(\s*[A-Z][a-z]+\s+(?:and|&)\s+[A-Z][a-z]+,?\s*\d{4}[a-z]?\s*\)'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text with fallback options"""
        if not text or not isinstance(text, str):
            return []
        
        sentences = []
        
        # Try NLTK first
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                # Filter out very short sentences
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                return sentences
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
        
        # Fallback: improved sentence splitting that preserves medical abbreviations
        # First, protect common medical abbreviations
        protected_abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'et al.', 'i.e.', 'e.g.', 'vs.', 'cf.']
        temp_replacements = {}
        
        for i, abbrev in enumerate(protected_abbrevs):
            placeholder = f"__ABBREV_{i}__"
            temp_replacements[placeholder] = abbrev
            text = text.replace(abbrev, placeholder)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviations
        for placeholder, abbrev in temp_replacements.items():
            sentences = [s.replace(placeholder, abbrev) for s in sentences]
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences


class MedicalEntityExtractor:
    """Extract medical entities from text"""
    
    def __init__(self):
        """Initialize medical entity extractor"""
        self.nlp = None
        self.medical_nlp = None
        self._models_loaded = False
        self._load_models()
        
        # Enhanced medical entity patterns
        self.entity_patterns = {
            'diseases': [
                r'\b(?:pneumonia|tuberculosis|cancer|diabetes|hypertension|asthma)\b',
                r'\b(?:covid-19|covid|sars-cov-2|sars|influenza|hepatitis|hiv|aids)\b',
                r'\b(?:stroke|myocardial\s+infarction|heart\s+attack|angina)\b',
                r'\b(?:alzheimer|parkinson|epilepsy|migraine|depression)\b',
                r'\b(?:copd|emphysema|bronchitis|pneumothorax)\b'
            ],
            'drugs': [
                r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|insulin|metformin)\b',
                r'\b(?:antibiotic|analgesic|antihypertensive|diuretic|statin)\b',
                r'\b[A-Z][a-z]+(?:cillin|mycin|cycline|prazole|sartan|pril|olol)\b'
            ],
            'anatomy': [
                r'\b(?:heart|lung|liver|kidney|brain|stomach|pancreas)\b',
                r'\b(?:artery|vein|vessel|ventricle|atrium|aorta)\b',
                r'\b(?:cortex|hippocampus|cerebellum|brainstem|thalamus)\b'
            ],
            'procedures': [
                r'\b(?:surgery|biopsy|endoscopy|angiography|catheterization)\b',
                r'\b(?:MRI|CT\s+scan|ultrasound|X-ray|PET\s+scan)\b'
            ]
        }
    
    def _load_models(self):
        """Load NLP models safely"""
        try:
            if SCISPACY_AVAILABLE:
                try:
                    import en_ner_bc5cdr_md
                    self.medical_nlp = spacy.load("en_ner_bc5cdr_md")
                    logger.info("Loaded scispacy medical NER model")
                except (ImportError, OSError) as e:
                    logger.warning(f"Failed to load scispacy medical NER model: {e}")
                    self.medical_nlp = None
        except Exception as e:
            logger.warning(f"Error loading medical NLP model: {e}")
            self.medical_nlp = None
        
        try:
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spacy English model")
                except (ImportError, OSError) as e:
                    logger.warning(f"Failed to load spacy English model: {e}")
                    self.nlp = None
        except Exception as e:
            logger.warning(f"Error loading general NLP model: {e}")
            self.nlp = None
        
        self._models_loaded = True
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their occurrences
        """
        if not text or not isinstance(text, str):
            return {key: [] for key in self.entity_patterns.keys()}
        
        entities = {key: [] for key in self.entity_patterns.keys()}
        
        # Use scispacy if available
        if self.medical_nlp:
            try:
                doc = self.medical_nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "DISEASE" and ent.text.strip():
                        entities['diseases'].append(ent.text.strip())
                    elif ent.label_ == "CHEMICAL" and ent.text.strip():
                        entities['drugs'].append(ent.text.strip())
            except Exception as e:
                logger.warning(f"Scispacy entity extraction failed: {e}")
        
        # Pattern-based extraction with error handling
        for entity_type, patterns in self.entity_patterns.items():
            pattern_entities = self._extract_by_pattern(text, patterns)
            entities[entity_type].extend(pattern_entities)
        
        # Remove duplicates and clean
        for entity_type in entities:
            # Convert to lowercase for deduplication, then back to original case
            seen = set()
            unique_entities = []
            for entity in entities[entity_type]:
                entity_clean = entity.strip().lower()
                if entity_clean and len(entity_clean) > 2 and entity_clean not in seen:
                    seen.add(entity_clean)
                    unique_entities.append(entity.strip())
            entities[entity_type] = unique_entities[:20]  # Limit to 20 per type
        
        return entities
    
    def _extract_by_pattern(self, text: str, patterns: List[str]) -> List[str]:
        """Extract entities using regex patterns with error handling"""
        entities = []
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities.extend([match.strip() for match in matches if match.strip()])
            except re.error as e:
                logger.warning(f"Regex error in pattern '{pattern}': {e}")
                continue
        return entities


class MedicalTextProcessor:
    """Medical domain-specific text processing"""
    
    def __init__(self):
        """Initialize medical text processor"""
        self.cleaner = TextCleaner()
        self.entity_extractor = MedicalEntityExtractor()
        
        # Expanded medical abbreviation dictionary
        self.medical_abbreviations = {
            'MI': 'myocardial infarction',
            'CHF': 'congestive heart failure',
            'COPD': 'chronic obstructive pulmonary disease',
            'HTN': 'hypertension',
            'DM': 'diabetes mellitus',
            'CAD': 'coronary artery disease',
            'CVA': 'cerebrovascular accident',
            'PE': 'pulmonary embolism',
            'DVT': 'deep vein thrombosis',
            'URI': 'upper respiratory infection',
            'GERD': 'gastroesophageal reflux disease',
            'UTI': 'urinary tract infection',
            'CHD': 'coronary heart disease',
            'ICU': 'intensive care unit',
            'ER': 'emergency room',
            'OR': 'operating room'
        }
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations safely"""
        if not text or not isinstance(text, str):
            return ""
        
        expanded_text = text
        for abbrev, full_form in self.medical_abbreviations.items():
            try:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(abbrev) + r'\b'
                replacement = f"{abbrev} ({full_form})"
                expanded_text = re.sub(pattern, replacement, expanded_text, flags=re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Error expanding abbreviation '{abbrev}': {e}")
                continue
        
        return expanded_text
    
    def extract_clinical_findings(self, text: str) -> List[str]:
        """Extract clinical findings from medical text"""
        if not text:
            return []
        
        findings_patterns = [
            r'patient\s+(?:presents|presented)\s+with\s+([^.]+)',
            r'(?:diagnosed|diagnosis)\s+(?:of|with)\s+([^.]+)',
            r'(?:symptoms|signs)\s+(?:include|included)\s+([^.]+)',
            r'(?:shows|showed|demonstrates|demonstrated)\s+([^.]+)',
            r'(?:findings|results)\s+(?:show|showed|indicate|indicated)\s+([^.]+)'
        ]
        
        findings = []
        for pattern in findings_patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                findings.extend([match.strip() for match in matches if match.strip()])
            except re.error as e:
                logger.warning(f"Error in clinical findings pattern: {e}")
                continue
        
        return findings[:10]  # Limit results
    
    def extract_medical_values(self, text: str) -> Dict[str, List[str]]:
        """Extract medical measurements and values"""
        if not text:
            return {'vital_signs': [], 'lab_values': [], 'measurements': [], 'dosages': []}
        
        values = {
            'vital_signs': [],
            'lab_values': [],
            'measurements': [],
            'dosages': []
        }
        
        # Define patterns with error handling
        pattern_groups = {
            'vital_signs': [
                r'blood\s+pressure:?\s*(\d+/\d+)',
                r'heart\s+rate:?\s*(\d+)\s*bpm',
                r'temperature:?\s*(\d+\.?\d*)\s*[°]?[FCfc]',
                r'oxygen\s+saturation:?\s*(\d+)%',
                r'respiratory\s+rate:?\s*(\d+)'
            ],
            'lab_values': [
                r'hemoglobin:?\s*(\d+\.?\d*)\s*g/dl',
                r'glucose:?\s*(\d+)\s*mg/dl',
                r'creatinine:?\s*(\d+\.?\d*)\s*mg/dl',
                r'white\s+blood\s+cell\s+count:?\s*(\d+,?\d*)',
                r'platelet\s+count:?\s*(\d+,?\d*)'
            ],
            'dosages': [
                r'(\d+\.?\d*\s*(?:mg|ml|g|μg|mcg|IU|units?))',
            ]
        }
        
        for value_type, patterns in pattern_groups.items():
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    values[value_type].extend([match for match in matches if match])
                except re.error as e:
                    logger.warning(f"Error in {value_type} pattern: {e}")
                    continue
        
        return values


class CitationExtractor:
    """Extract and parse medical citations"""
    
    def __init__(self):
        """Initialize citation extractor"""
        # Improved citation patterns with better error handling
        self.citation_patterns = [
            # Author-year inline citations
            r'\(([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+(?:et\s+al\.?)?\s*,?\s*(\d{4})\)',
            
            # Numbered citations
            r'\[(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]',
            
            # DOI patterns
            r'doi:\s*([10]\.\d+/[^\s]+)',
        ]
    
    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract citations from text safely
        
        Args:
            text: Input text containing citations
            
        Returns:
            List of extracted citation dictionaries
        """
        if not text or not isinstance(text, str):
            return []
        
        citations = []
        
        for i, pattern in enumerate(self.citation_patterns):
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    
                    citation = {
                        'raw_text': match.group(0),
                        'position': match.span(),
                        'pattern_type': i,
                        'groups': groups
                    }
                    
                    # Parse based on pattern type
                    if i == 0 and len(groups) >= 2:  # Author-year
                        citation.update({
                            'authors': groups[0],
                            'year': groups[1],
                            'type': 'author_year'
                        })
                    elif i == 1:  # Numbered
                        citation.update({
                            'reference_numbers': groups[0],
                            'type': 'numbered'
                        })
                    elif i == 2:  # DOI
                        citation.update({
                            'doi': groups[0],
                            'type': 'doi'
                        })
                    
                    citations.append(citation)
                    
            except re.error as e:
                logger.warning(f"Citation pattern {i} failed: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error in citation extraction: {e}")
                continue
        
        return citations[:50]  # Limit to 50 citations
    
    def parse_reference_list(self, reference_text: str) -> List[Dict[str, Any]]:
        """Parse reference list section safely"""
        if not reference_text:
            return []
        
        references = []
        
        try:
            # Split by reference numbers with improved pattern
            ref_pattern = r'(?:^|\n)\s*(\d+)\.?\s+([^\n\d]+(?:\n(?!\s*\d+\.)[^\n]+)*)'
            matches = re.findall(ref_pattern, reference_text, re.MULTILINE)
            
            for number, ref_text in matches:
                try:
                    parsed_ref = self._parse_single_reference(ref_text.strip())
                    parsed_ref['number'] = int(number)
                    references.append(parsed_ref)
                except ValueError as e:
                    logger.warning(f"Error parsing reference {number}: {e}")
                    continue
                    
        except re.error as e:
            logger.error(f"Reference list parsing failed: {e}")
        
        return references[:100]  # Limit to 100 references
    
    def _parse_single_reference(self, ref_text: str) -> Dict[str, Any]:
        """Parse a single reference with error handling"""
        reference = {
            'raw_text': ref_text,
            'authors': '',
            'title': '',
            'journal': '',
            'year': '',
            'volume': '',
            'pages': '',
            'doi': '',
            'type': 'journal'
        }
        
        if not ref_text:
            return reference
        
        try:
            # Extract DOI
            doi_match = re.search(r'doi:\s*([^\s]+)', ref_text, re.IGNORECASE)
            if doi_match:
                reference['doi'] = doi_match.group(1)
            
            # Extract year
            year_match = re.search(r'\b(\d{4})\b', ref_text)
            if year_match:
                reference['year'] = year_match.group(1)
            
            # Extract authors (first part before year)
            author_patterns = [
                r'^([^.]{1,200}?)(?:\s+\(\d{4}\)|\s+\d{4}|\.)',
                r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)*(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)*(?:\s+[A-Z][a-z]+)*)*)'
            ]
            
            for pattern in author_patterns:
                try:
                    author_match = re.search(pattern, ref_text)
                    if author_match:
                        reference['authors'] = author_match.group(1).strip(' .,')
                        break
                except re.error:
                    continue
        
        except Exception as e:
            logger.warning(f"Error parsing reference components: {e}")
        
        return reference


class LiteratureProcessor:
    """Main class for processing medical literature with improved error handling"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Initialize literature processor with validation
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
        """
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self.text_cleaner = TextCleaner()
        self.text_processor = MedicalTextProcessor()
        self.citation_extractor = CitationExtractor()
        
        # Initialize NLTK resources if available
        if NLTK_AVAILABLE:
            self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK resources safely"""
        try:
            required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for data_name in required_data:
                try:
                    nltk.download(data_name, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK data '{data_name}': {e}")
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
    
    def process_text(self, 
                    text: str,
                    title: str = "",
                    metadata: Optional[Dict[str, Any]] = None) -> ProcessedDocument:
        """
        Process medical literature text with comprehensive error handling
        
        Args:
            text: Input text
            title: Document title
            metadata: Additional metadata
            
        Returns:
            ProcessedDocument object
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 50:
            logger.warning("Text is empty, invalid, or too short for processing")
            return self._create_empty_document(title)
        
        try:
            # Clean text
            cleaned_text = self.text_cleaner.clean_text(text)
            if not cleaned_text:
                logger.warning("Text cleaning resulted in empty text")
                return self._create_empty_document(title)
            
            # Extract abstract if present
            abstract = self._extract_abstract(cleaned_text)
            
            # Expand abbreviations
            expanded_text = self.text_processor.expand_abbreviations(cleaned_text)
            
            # Extract entities with error handling
            try:
                entities = self.text_processor.entity_extractor.extract_entities(expanded_text)
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
                entities = {}
            
            # Extract citations with error handling
            try:
                citations = self.citation_extractor.extract_citations(expanded_text)
            except Exception as e:
                logger.warning(f"Citation extraction failed: {e}")
                citations = []
            
            # Create chunks with error handling
            try:
                chunks = self.create_chunks(expanded_text)
            except Exception as e:
                logger.warning(f"Text chunking failed: {e}")
                chunks = []
            
            # Extract keywords
            try:
                keywords = self._extract_keywords(expanded_text, entities)
            except Exception as e:
                logger.warning(f"Keyword extraction failed: {e}")
                keywords = []
            
            # Create processed document
            processed_doc = ProcessedDocument(
                title=title or self._extract_title(text),
                abstract=abstract,
                full_text=expanded_text,
                chunks=chunks,
                metadata=metadata or {},
                entities=entities,
                citations=citations,
                keywords=keywords,
                processed_at=datetime.now().isoformat()
            )
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return self._create_empty_document(title)
    
    def process_file(self, 
                    file_path: str,
                    encoding: str = 'utf-8') -> Optional[ProcessedDocument]:
        """
        Process literature file with improved error handling
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Returns:
            ProcessedDocument or None if processing fails
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            if not path.is_file():
                logger.error(f"Path is not a file: {file_path}")
                return None
            
            # Initialize variables
            text = ""
            metadata = {}
            
            # Read file based on extension
            try:
                if path.suffix.lower() == '.pdf':
                    if PDF_EXTRACTOR_AVAILABLE:
                        extractor = PDFExtractor()
                        text = extractor.extract_text(file_path)
                        pdf_metadata = extractor.extract_metadata(file_path)
                        
                        # Convert PDFMetadata to dict safely
                        if pdf_metadata and hasattr(pdf_metadata, '__dict__'):
                            metadata = {
                                'file_path': str(path),
                                'file_size': pdf_metadata.file_size,
                                'pages': pdf_metadata.pages,
                                'title': getattr(pdf_metadata, 'title', ''),
                                'author': getattr(pdf_metadata, 'author', ''),
                                'doi': getattr(pdf_metadata, 'doi', ''),
                                'journal': getattr(pdf_metadata, 'journal', '')
                            }
                    else:
                        logger.error("PDF extractor not available for PDF files")
                        return None
                else:
                    # Handle text files
                    try:
                        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                            text = f.read()
                        metadata = {
                            'file_path': str(path), 
                            'file_size': path.stat().st_size,
                            'encoding': encoding
                        }
                    except UnicodeDecodeError:
                        # Try different encodings
                        for alt_encoding in ['latin-1', 'cp1252', 'utf-8-sig']:
                            try:
                                with open(file_path, 'r', encoding=alt_encoding, errors='ignore') as f:
                                    text = f.read()
                                metadata['encoding'] = alt_encoding
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            logger.error(f"Could not decode file with any encoding: {file_path}")
                            return None
            
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return None
            
            if not text or len(text.strip()) < 10:
                logger.error(f"No meaningful text extracted from {file_path}")
                return None
            
            # Process the text
            title = path.stem.replace('_', ' ').replace('-', ' ').title()
            return self.process_text(text, title=title, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {e}")
            return None
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Create text chunks for RAG pipeline with improved error handling
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
        
        try:
            # Split into sentences
            sentences = self.text_cleaner.extract_sentences(text)
            
            if not sentences:
                logger.warning("No sentences found in text")
                return []
            
            chunks = []
            current_chunk = ""
            current_size = 0
            
            for sentence in sentences:
                if not sentence:  # Skip empty sentences
                    continue
                    
                sentence_size = len(sentence)
                
                # If adding this sentence would exceed chunk size
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    if current_size >= self.min_chunk_size:
                        chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and chunks:
                        try:
                            overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                            current_chunk = f"{overlap_text} {sentence}" if overlap_text else sentence
                            current_size = len(current_chunk)
                        except Exception as e:
                            logger.warning(f"Overlap calculation failed: {e}")
                            current_chunk = sentence
                            current_size = sentence_size
                    else:
                        current_chunk = sentence
                        current_size = sentence_size
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_size = len(current_chunk)
            
            # Add final chunk
            if current_chunk and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
            
            logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
            return chunks
            
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            return []
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get text for chunk overlap with bounds checking"""
        if not text or overlap_size <= 0:
            return ""
        
        if len(text) <= overlap_size:
            return text
        
        try:
            # Find sentence boundaries near the overlap point
            sentences = self.text_cleaner.extract_sentences(text)
            
            if not sentences:
                # Fallback: take last overlap_size characters
                return text[-overlap_size:].strip()
            
            overlap_text = ""
            for sentence in reversed(sentences):
                if not sentence:
                    continue
                if len(overlap_text) + len(sentence) + 1 <= overlap_size:  # +1 for space
                    if overlap_text:
                        overlap_text = sentence + " " + overlap_text
                    else:
                        overlap_text = sentence
                else:
                    break
            
            return overlap_text.strip()
            
        except Exception as e:
            logger.warning(f"Overlap text calculation failed: {e}")
            # Fallback: simple truncation
            return text[-overlap_size:].strip()
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from text with improved patterns"""
        if not text:
            return ""
        
        abstract_patterns = [
            r'(?:^|\n)\s*abstract\s*[:\n]\s*(.*?)(?=\n\s*(?:introduction|keywords|methods|\d+\.|\Z))',
            r'(?:^|\n)\s*summary\s*[:\n]\s*(.*?)(?=\n\s*(?:introduction|keywords|methods|\d+\.|\Z))',
            r'(?:^|\n)\s*abstract\s*\n(.*?)(?=\n\s*[A-Z][A-Z\s]{3,})',
        ]
        
        for pattern in abstract_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1).strip()
                    # Clean up the abstract
                    abstract = re.sub(r'\s+', ' ', abstract)
                    if 50 <= len(abstract) <= 2000:  # Reasonable abstract length
                        return abstract
            except re.error as e:
                logger.warning(f"Abstract extraction pattern failed: {e}")
                continue
        
        # If no abstract found, return first substantial paragraph
        try:
            paragraphs = text.split('\n\n')
            for para in paragraphs[:5]:  # Check first 5 paragraphs only
                para = para.strip()
                if (100 <= len(para) <= 1000 and 
                    not para.lower().startswith(('table', 'figure', 'reference', 'keyword'))):
                    return para
        except Exception as e:
            logger.warning(f"Fallback abstract extraction failed: {e}")
        
        return ""
    
    def _extract_title(self, text: str) -> str:
        """Extract title from text with improved heuristics"""
        if not text:
            return "Untitled Document"
        
        try:
            lines = text.split('\n')
            
            for line in lines[:15]:  # Check first 15 lines
                line = line.strip()
                if not line:
                    continue
                    
                # Skip common non-title patterns
                if line.lower().startswith(('abstract', 'introduction', 'method', 'background', 
                                         'table', 'figure', 'page', 'doi:', 'http')):
                    continue
                
                # Check if it looks like a title
                if (10 <= len(line) <= 200 and  # Reasonable title length
                    len(line.split()) >= 3 and  # At least 3 words
                    not line.endswith('.') and  # Titles usually don't end with period
                    any(c.isupper() for c in line)):  # Has some uppercase
                    return line[:200]  # Limit title length
            
            return "Untitled Document"
            
        except Exception as e:
            logger.warning(f"Title extraction failed: {e}")
            return "Untitled Document"
    
    def _extract_keywords(self, text: str, entities: Dict[str, List[str]] = None) -> List[str]:
        """Extract keywords from text with multiple strategies"""
        if not text:
            return []
        
        keywords = []
        
        try:
            # Look for explicit keywords section
            keyword_patterns = [
                r'keywords?\s*[:\-]\s*(.*?)(?=\n\s*[A-Z]|\Z)',
                r'key\s*words?\s*[:\-]\s*(.*?)(?=\n\s*[A-Z]|\Z)',
                r'index\s*terms?\s*[:\-]\s*(.*?)(?=\n\s*[A-Z]|\Z)'
            ]
            
            for pattern in keyword_patterns:
                try:
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        keyword_text = match.group(1)
                        # Split by common separators
                        kw_list = re.split(r'[;,\n]', keyword_text)
                        keywords.extend([kw.strip().lower() for kw in kw_list 
                                       if kw.strip() and len(kw.strip()) > 2])
                        break
                except re.error:
                    continue
            
            # If no explicit keywords, extract from entities
            if not keywords and entities:
                for entity_list in entities.values():
                    if entity_list:
                        keywords.extend([e.lower() for e in entity_list])
            
            # Remove duplicates and filter
            keywords = list(dict.fromkeys(keywords))  # Preserve order while removing duplicates
            keywords = [kw for kw in keywords if 2 < len(kw) < 50]  # Filter by length
            
            return keywords[:25]  # Limit to top 25 keywords
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def _create_empty_document(self, title: str = "") -> ProcessedDocument:
        """Create empty processed document safely"""
        return ProcessedDocument(
            title=title or "Empty Document",
            abstract="",
            full_text="",
            chunks=[],
            metadata={},
            entities={},
            citations=[],
            keywords=[],
            processed_at=datetime.now().isoformat()
        )


# Convenience functions with improved error handling
def process_medical_literature(file_path: str, **kwargs) -> Optional[ProcessedDocument]:
    """
    Convenience function to process medical literature file
    
    Args:
        file_path: Path to literature file
        **kwargs: Additional parameters for LiteratureProcessor
        
    Returns:
        ProcessedDocument or None
    """
    try:
        processor = LiteratureProcessor(**kwargs)
        return processor.process_file(file_path)
    except Exception as e:
        logger.error(f"Literature processing failed: {e}")
        return None


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Convenience function to extract medical entities
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of extracted entities
    """
    try:
        extractor = MedicalEntityExtractor()
        return extractor.extract_entities(text)
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {}


def chunk_medical_text(text: str, 
                      chunk_size: int = 512,
                      chunk_overlap: int = 50) -> List[str]:
    """
    Convenience function to chunk medical text
    
    Args:
        text: Input text
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        processor = LiteratureProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return processor.create_chunks(text)
    except Exception as e:
        logger.error(f"Text chunking failed: {e}")
        return []


def create_literature_embeddings(documents: List[ProcessedDocument],
                                embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Create embeddings for processed literature documents with error handling
    
    Args:
        documents: List of processed documents
        embedding_model: Model to use for embeddings
        
    Returns:
        Dictionary with embeddings and metadata
    """
    if not documents:
        logger.warning("No documents provided for embedding creation")
        return {}
    
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers library not available for embedding creation")
        return {}
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(embedding_model)
        
        embeddings_data = {
            'document_embeddings': [],
            'chunk_embeddings': [],
            'metadata': [],
            'model': embedding_model,
            'created_at': datetime.now().isoformat()
        }
        
        for doc in documents:
            try:
                # Create document-level embedding from title + abstract
                doc_text = f"{doc.title} {doc.abstract}".strip()
                if doc_text and len(doc_text) > 10:
                    doc_embedding = model.encode(doc_text)
                    embeddings_data['document_embeddings'].append({
                        'embedding': doc_embedding.tolist(),
                        'text': doc_text,
                        'title': doc.title,
                        'type': 'document'
                    })
                
                # Create chunk embeddings
                for i, chunk in enumerate(doc.chunks):
                    if chunk and len(chunk.strip()) > 20:  # Only process meaningful chunks
                        try:
                            chunk_embedding = model.encode(chunk)
                            embeddings_data['chunk_embeddings'].append({
                                'embedding': chunk_embedding.tolist(),
                                'text': chunk,
                                'document_title': doc.title,
                                'chunk_index': i,
                                'type': 'chunk'
                            })
                        except Exception as e:
                            logger.warning(f"Failed to create embedding for chunk {i}: {e}")
                            continue
                
                # Store document metadata safely
                embeddings_data['metadata'].append({
                    'title': doc.title,
                    'entities': doc.entities,
                    'keywords': doc.keywords,
                    'citation_count': len(doc.citations),
                    'chunk_count': len(doc.chunks),
                    'processed_at': doc.processed_at
                })
                
            except Exception as e:
                logger.warning(f"Failed to process document '{doc.title}': {e}")
                continue
        
        logger.info(f"Created embeddings for {len(documents)} documents")
        return embeddings_data
        
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        return {}


class LiteraturePipeline:
    """Complete literature processing pipeline with robust error handling"""
    
    def __init__(self, 
                 output_dir: str = "data/processed/literature",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize literature processing pipeline
        
        Args:
            output_dir: Directory to save processed literature
            embedding_model: Model for creating embeddings
        """
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            raise
        
        self.embedding_model = embedding_model
        self.processor = LiteratureProcessor()
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.total_chunks = 0
        self.errors = []
    
    def process_directory(self, 
                         input_dir: str,
                         file_extensions: List[str] = ['.txt', '.pdf'],
                         save_processed: bool = True,
                         create_embeddings: bool = True) -> List[ProcessedDocument]:
        """
        Process all literature files in a directory with comprehensive error handling
        
        Args:
            input_dir: Directory containing literature files
            file_extensions: File extensions to process
            save_processed: Whether to save processed documents
            create_embeddings: Whether to create embeddings
            
        Returns:
            List of processed documents
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            error_msg = f"Input directory not found: {input_dir}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return []
        
        # Find all files with error handling
        files = []
        try:
            for ext in file_extensions:
                files.extend(list(input_path.rglob(f"*{ext}")))
        except Exception as e:
            error_msg = f"Error finding files in directory: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return []
        
        if not files:
            logger.warning(f"No files found with extensions {file_extensions} in {input_dir}")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        
        processed_docs = []
        
        for file_path in files:
            logger.info(f"Processing: {file_path.name}")
            
            try:
                doc = self.processor.process_file(str(file_path))
                if doc and doc.full_text:  # Ensure document has content
                    processed_docs.append(doc)
                    self.processed_count += 1
                    self.total_chunks += len(doc.chunks)
                    
                    # Save processed document
                    if save_processed:
                        try:
                            self._save_processed_document(doc)
                        except Exception as e:
                            logger.warning(f"Failed to save document {file_path.name}: {e}")
                else:
                    self.failed_count += 1
                    self.errors.append(f"No content extracted from {file_path.name}")
                    
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {e}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                self.failed_count += 1
        
        # Create embeddings if requested
        if create_embeddings and processed_docs:
            try:
                self._create_and_save_embeddings(processed_docs)
            except Exception as e:
                logger.error(f"Failed to create embeddings: {e}")
                self.errors.append(f"Embedding creation failed: {e}")
        
        logger.info(f"Processing complete: {self.processed_count} successful, {self.failed_count} failed")
        return processed_docs
    
    def _save_processed_document(self, doc: ProcessedDocument):
        """Save processed document to JSON with error handling"""
        try:
            # Sanitize filename
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', doc.title)
            safe_title = safe_title[:100]  # Limit filename length
            filename = f"{safe_title}.json"
            filepath = self.output_dir / filename
            
            # Convert to serializable format
            doc_dict = doc.to_dict()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            raise
    
    def _create_and_save_embeddings(self, documents: List[ProcessedDocument]):
        """Create and save embeddings for documents"""
        logger.info("Creating embeddings...")
        
        try:
            embeddings_data = create_literature_embeddings(documents, self.embedding_model)
            
            if embeddings_data:
                embeddings_file = self.output_dir / "embeddings.json"
                with open(embeddings_file, 'w') as f:
                    json.dump(embeddings_data, f, indent=2)
                
                logger.info(f"Saved embeddings to {embeddings_file}")
            else:
                logger.warning("No embeddings created")
        except Exception as e:
            logger.error(f"Embedding creation and saving failed: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'processed_documents': self.processed_count,
            'failed_documents': self.failed_count,
            'total_chunks': self.total_chunks,
            'average_chunks_per_doc': self.total_chunks / max(self.processed_count, 1),
            'output_directory': str(self.output_dir),
            'error_count': len(self.errors),
            'errors': self.errors[-10:]  # Last 10 errors
        }


# Export all classes and functions
__all__ = [
    'LiteratureProcessor',
    'TextCleaner',
    'MedicalTextProcessor',
    'MedicalEntityExtractor',
    'CitationExtractor',
    'ProcessedDocument',
    'LiteraturePipeline',
    'process_medical_literature',
    'extract_medical_entities',
    'chunk_medical_text',
    'create_literature_embeddings'
]


# Configuration presets
class ProcessingConfig:
    """Processing configuration presets with validation"""
    
    BASIC = {
        'chunk_size': 256,
        'chunk_overlap': 25,
        'min_chunk_size': 50
    }
    
    DETAILED = {
        'chunk_size': 512,
        'chunk_overlap': 50,
        'min_chunk_size': 100
    }
    
    COMPREHENSIVE = {
        'chunk_size': 1024,
        'chunk_overlap': 100,
        'min_chunk_size': 200
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """Validate configuration parameters"""
        errors = []
        
        if config.get('chunk_size', 0) <= 0:
            errors.append("chunk_size must be positive")
        if config.get('chunk_overlap', -1) < 0:
            errors.append("chunk_overlap cannot be negative")
        if config.get('min_chunk_size', 0) <= 0:
            errors.append("min_chunk_size must be positive")
        if config.get('chunk_overlap', 0) >= config.get('chunk_size', 1):
            errors.append("chunk_overlap must be less than chunk_size")
            
        return errors


def batch_process_literature(input_directories: List[str],
                           output_dir: str = "data/processed/literature",
                           config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process literature from multiple directories with comprehensive error handling
    
    Args:
        input_directories: List of directories to process
        output_dir: Output directory
        config: Processing configuration
        
    Returns:
        Processing results summary
    """
    if not input_directories:
        return {'error': 'No input directories provided'}
    
    # Validate configuration
    if config is None:
        config = ProcessingConfig.DETAILED
    
    config_errors = ProcessingConfig.validate_config(config)
    if config_errors:
        return {'error': f'Invalid configuration: {"; ".join(config_errors)}'}
    
    try:
        pipeline = LiteraturePipeline(output_dir=output_dir)
        pipeline.processor = LiteratureProcessor(**config)
        
        all_documents = []
        directory_stats = {}
        
        for input_dir in input_directories:
            logger.info(f"Processing directory: {input_dir}")
            
            try:
                docs = pipeline.process_directory(
                    input_dir,
                    save_processed=True,
                    create_embeddings=False  # Create embeddings at the end
                )
                
                all_documents.extend(docs)
                directory_stats[input_dir] = {
                    'processed': len(docs),
                    'failed': pipeline.failed_count
                }
                
            except Exception as e:
                logger.error(f"Failed to process directory {input_dir}: {e}")
                directory_stats[input_dir] = {
                    'processed': 0,
                    'failed': 0,
                    'error': str(e)
                }
        
        # Create embeddings for all documents
        if all_documents:
            try:
                pipeline._create_and_save_embeddings(all_documents)
            except Exception as e:
                logger.error(f"Failed to create final embeddings: {e}")
        
        return {
            'total_documents': len(all_documents),
            'directory_breakdown': directory_stats,
            'pipeline_stats': pipeline.get_statistics(),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return {
            'error': str(e),
            'success': False
        }