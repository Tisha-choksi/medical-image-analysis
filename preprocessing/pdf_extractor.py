"""
PDF Extraction Module for Medical Literature
==========================================

This module provides PDF text extraction capabilities specifically designed for medical literature.
Handles various PDF formats, structures, and layouts commonly found in medical papers.

Classes:
- PDFExtractor: Main PDF extraction class
- MedicalPDFProcessor: Medical domain-specific PDF processing
- PDFStructureParser: Parse structured elements from PDFs

Usage:
    from preprocessing.pdf_extractor import PDFExtractor
    
    extractor = PDFExtractor()
    text = extractor.extract_text('medical_paper.pdf')
    metadata = extractor.extract_metadata('medical_paper.pdf')
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import tempfile

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfminer
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PDFMetadata:
    """Container for PDF metadata"""
    title: str
    author: str
    subject: str
    creator: str
    producer: str
    creation_date: str
    modification_date: str
    pages: int
    encrypted: bool
    file_size: int
    doi: str
    journal: str
    keywords: List[str]


@dataclass
class ExtractedContent:
    """Container for extracted PDF content"""
    text: str
    metadata: PDFMetadata
    sections: Dict[str, str]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    references: List[str]
    page_texts: List[str]


class PDFExtractor:
    """Main PDF extraction class with fallback mechanisms"""
    
    def __init__(self, 
                 prefer_pdfplumber: bool = True,
                 enable_ocr: bool = False,
                 ocr_threshold: float = 0.5):
        """
        Initialize PDF extractor
        
        Args:
            prefer_pdfplumber: Use pdfplumber as primary extractor
            enable_ocr: Enable OCR for scanned PDFs
            ocr_threshold: Threshold for OCR confidence
        """
        self.prefer_pdfplumber = prefer_pdfplumber
        self.enable_ocr = enable_ocr
        self.ocr_threshold = ocr_threshold
        
        # Check available libraries
        self.available_extractors = []
        if PDFPLUMBER_AVAILABLE:
            self.available_extractors.append('pdfplumber')
        if PYPDF2_AVAILABLE:
            self.available_extractors.append('pypdf2')
        if PYMUPDF_AVAILABLE:
            self.available_extractors.append('pymupdf')
        if PDFMINER_AVAILABLE:
            self.available_extractors.append('pdfminer')
        
        if not self.available_extractors:
            raise ImportError("No PDF extraction libraries available. Install PyPDF2, pdfplumber, PyMuPDF, or pdfminer")
        
        logger.info(f"Available PDF extractors: {', '.join(self.available_extractors)}")
    
    def extract_text(self, pdf_path: str, method: str = 'auto') -> str:
        """
        Extract text from PDF using specified or automatic method selection
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('auto', 'pdfplumber', 'pypdf2', 'pymupdf', 'pdfminer')
            
        Returns:
            Extracted text
        """
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return ""
        
        # Auto-select method
        if method == 'auto':
            method = self._select_best_method(pdf_path)
        
        try:
            if method == 'pdfplumber' and PDFPLUMBER_AVAILABLE:
                return self._extract_with_pdfplumber(pdf_path)
            elif method == 'pypdf2' and PYPDF2_AVAILABLE:
                return self._extract_with_pypdf2(pdf_path)
            elif method == 'pymupdf' and PYMUPDF_AVAILABLE:
                return self._extract_with_pymupdf(pdf_path)
            elif method == 'pdfminer' and PDFMINER_AVAILABLE:
                return self._extract_with_pdfminer(pdf_path)
            else:
                # Fallback to first available method
                return self._extract_with_fallback(pdf_path)
                
        except Exception as e:
            logger.error(f"Text extraction failed with {method}: {str(e)}")
            return self._extract_with_fallback(pdf_path)
    
    def _select_best_method(self, pdf_path: str) -> str:
        """Select best extraction method based on PDF characteristics"""
        try:
            # Quick analysis to determine best method
            if PDFPLUMBER_AVAILABLE:
                with pdfplumber.open(pdf_path) as pdf:
                    first_page = pdf.pages[0] if pdf.pages else None
                    if first_page:
                        # Check if PDF has tables or complex layout
                        tables = first_page.extract_tables()
                        if tables:
                            return 'pdfplumber'  # Best for tables
                        
                        # Check text density
                        text = first_page.extract_text()
                        if text and len(text.strip()) > 100:
                            return 'pdfplumber'
            
            # Default preference order
            if self.prefer_pdfplumber and 'pdfplumber' in self.available_extractors:
                return 'pdfplumber'
            elif 'pymupdf' in self.available_extractors:
                return 'pymupdf'  # Fast and reliable
            elif 'pypdf2' in self.available_extractors:
                return 'pypdf2'
            else:
                return self.available_extractors[0]
                
        except Exception as e:
            logger.warning(f"Method selection failed: {str(e)}")
            return self.available_extractors[0]
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return '\n\n'.join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract page text: {str(e)}")
                    continue
        
        return '\n\n'.join(text_parts)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        text_parts = []
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
        
        doc.close()
        return '\n\n'.join(text_parts)
    
    def _extract_with_pdfminer(self, pdf_path: str) -> str:
        """Extract text using pdfminer"""
        try:
            # Configure layout analysis parameters for better text extraction
            laparams = LAParams(
                boxes_flow=0.5,
                word_margin=0.1,
                char_margin=2.0,
                line_margin=0.5
            )
            
            text = pdfminer_extract_text(pdf_path, laparams=laparams)
            return text
        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {str(e)}")
            return ""
    
    def _extract_with_fallback(self, pdf_path: str) -> str:
        """Try all available methods as fallback"""
        for method in self.available_extractors:
            try:
                if method == 'pdfplumber':
                    return self._extract_with_pdfplumber(pdf_path)
                elif method == 'pypdf2':
                    return self._extract_with_pypdf2(pdf_path)
                elif method == 'pymupdf':
                    return self._extract_with_pymupdf(pdf_path)
                elif method == 'pdfminer':
                    return self._extract_with_pdfminer(pdf_path)
            except Exception as e:
                logger.warning(f"Fallback {method} failed: {str(e)}")
                continue
        
        logger.error("All extraction methods failed")
        return ""
    
    def extract_metadata(self, pdf_path: str) -> PDFMetadata:
        """
        Extract metadata from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFMetadata object
        """
        metadata = PDFMetadata(
            title="", author="", subject="", creator="", producer="",
            creation_date="", modification_date="", pages=0, encrypted=False,
            file_size=0, doi="", journal="", keywords=[]
        )
        
        try:
            file_size = Path(pdf_path).stat().st_size
            metadata.file_size = file_size
            
            # Try PyPDF2 for metadata (most reliable)
            if PYPDF2_AVAILABLE:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    metadata.pages = len(pdf_reader.pages)
                    metadata.encrypted = pdf_reader.is_encrypted
                    
                    if pdf_reader.metadata:
                        info = pdf_reader.metadata
                        metadata.title = str(info.get('/Title', '')) if info.get('/Title') else ''
                        metadata.author = str(info.get('/Author', '')) if info.get('/Author') else ''
                        metadata.subject = str(info.get('/Subject', '')) if info.get('/Subject') else ''
                        metadata.creator = str(info.get('/Creator', '')) if info.get('/Creator') else ''
                        metadata.producer = str(info.get('/Producer', '')) if info.get('/Producer') else ''
                        
                        # Handle dates
                        if info.get('/CreationDate'):
                            metadata.creation_date = str(info.get('/CreationDate'))
                        if info.get('/ModDate'):
                            metadata.modification_date = str(info.get('/ModDate'))
            
            # Extract additional metadata from text content
            text_content = self.extract_text(pdf_path)
            if text_content:
                # Extract DOI
                doi_match = re.search(r'doi:?\s*(10\.\d+/[^\s]+)', text_content, re.IGNORECASE)
                if doi_match:
                    metadata.doi = doi_match.group(1)
                
                # Extract journal name
                journal_patterns = [
                    r'published\s+in\s+([^,\n]+)',
                    r'journal\s+of\s+([^,\n]+)',
                    r'^\s*([A-Z][^,\n]*(?:Journal|Review|Medicine|Science)[^,\n]*)',
                ]
                
                for pattern in journal_patterns:
                    match = re.search(pattern, text_content, re.IGNORECASE | re.MULTILINE)
                    if match:
                        metadata.journal = match.group(1).strip()
                        break
                
                # Extract keywords
                keywords_match = re.search(r'keywords?:?\s*([^\n]+)', text_content, re.IGNORECASE)
                if keywords_match:
                    keywords_text = keywords_match.group(1)
                    metadata.keywords = [kw.strip() for kw in re.split(r'[;,]', keywords_text) if kw.strip()]
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return metadata
    
    def extract_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text using OCR for scanned PDFs
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            OCR extracted text
        """
        if not TESSERACT_AVAILABLE or not PYMUPDF_AVAILABLE:
            logger.error("OCR requires pytesseract and PyMuPDF")
            return ""
        
        try:
            text_parts = []
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Try regular text extraction first
                page_text = page.get_text()
                
                # If minimal text found, use OCR
                if not page_text or len(page_text.strip()) < 50:
                    # Convert page to image
                    mat = fitz.Matrix(2, 2)  # Scale factor for better OCR
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_file.write(img_data)
                        temp_path = temp_file.name
                    
                    try:
                        # Perform OCR
                        ocr_text = pytesseract.image_to_string(Image.open(temp_path))
                        if ocr_text and len(ocr_text.strip()) > 20:
                            text_parts.append(ocr_text)
                        else:
                            text_parts.append(page_text)  # Use regular text if OCR fails
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    text_parts.append(page_text)
            
            doc.close()
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return ""


class MedicalPDFProcessor:
    """Medical domain-specific PDF processing"""
    
    def __init__(self):
        """Initialize medical PDF processor"""
        self.extractor = PDFExtractor()
        
        # Medical journal section patterns
        self.section_patterns = {
            'abstract': r'(?:^|\n)\s*abstract\s*\n(.*?)(?=\n\s*(?:introduction|keywords|methods|\d+\.|$))',
            'introduction': r'(?:^|\n)\s*(?:\d+\.?\s*)?introduction\s*\n(.*?)(?=\n\s*(?:\d+\.|\w+:|\Z))',
            'methods': r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:methods|methodology|materials and methods)\s*\n(.*?)(?=\n\s*(?:\d+\.|\w+:|\Z))',
            'results': r'(?:^|\n)\s*(?:\d+\.?\s*)?results\s*\n(.*?)(?=\n\s*(?:\d+\.|\w+:|\Z))',
            'discussion': r'(?:^|\n)\s*(?:\d+\.?\s*)?discussion\s*\n(.*?)(?=\n\s*(?:\d+\.|\w+:|\Z))',
            'conclusion': r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:conclusion|conclusions)\s*\n(.*?)(?=\n\s*(?:\d+\.|\w+:|\Z))',
            'references': r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:references|bibliography)\s*\n(.*?)(?=\n\s*(?:\d+\.|\w+:|\Z))',
            'acknowledgments': r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:acknowledgments?|acknowledgements?)\s*\n(.*?)(?=\n\s*(?:\d+\.|\w+:|\Z))'
        }
        
        # Medical terminology patterns
        self.medical_patterns = {
            'dosages': r'\b\d+\s*(?:mg|ml|g|μg|mcg|IU|units?)\b',
            'measurements': r'\b\d+\.?\d*\s*(?:cm|mm|inches?|ft)\b',
            'percentages': r'\b\d+\.?\d*%\b',
            'p_values': r'\bp\s*[<>=]\s*0\.\d+\b',
            'confidence_intervals': r'\b95%\s*CI\b',
            'sample_sizes': r'\bn\s*=\s*\d+\b'
        }
    
    def extract_structured_content(self, pdf_path: str) -> ExtractedContent:
        """
        Extract structured content from medical PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractedContent object with structured data
        """
        # Extract basic text and metadata
        text = self.extractor.extract_text(pdf_path)
        metadata = self.extractor.extract_metadata(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return self._create_empty_content(metadata)
        
        # Extract sections
        sections = self._extract_sections(text)
        
        # Extract tables and figures
        tables = self._extract_tables(pdf_path)
        figures = self._extract_figures(pdf_path)
        
        # Extract references
        references = self._extract_references(text, sections.get('references', ''))
        
        # Split text by pages
        page_texts = self._extract_page_texts(pdf_path)
        
        return ExtractedContent(
            text=text,
            metadata=metadata,
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            page_texts=page_texts
        )
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract structured sections from medical paper"""
        sections = {}
        
        for section_name, pattern in self.section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1).strip()
                # Clean up section text
                section_text = re.sub(r'\s+', ' ', section_text)
                sections[section_name] = section_text
        
        return sections
    
    def _extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        tables = []
        
        if not PDFPLUMBER_AVAILABLE:
            return tables
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Must have header and at least one row
                            table_data = {
                                'page': page_num + 1,
                                'table_number': table_num + 1,
                                'headers': table[0] if table[0] else [],
                                'rows': table[1:],
                                'row_count': len(table) - 1,
                                'col_count': len(table[0]) if table[0] else 0
                            }
                            
                            # Extract table caption if nearby
                            table_data['caption'] = self._find_table_caption(page, table_num)
                            
                            tables.append(table_data)
        
        except Exception as e:
            logger.warning(f"Table extraction failed: {str(e)}")
        
        return tables
    
    def _find_table_caption(self, page, table_num: int) -> str:
        """Find table caption near table"""
        try:
            # Extract all text from page
            page_text = page.extract_text()
            if not page_text:
                return ""
            
            # Look for table captions
            caption_patterns = [
                rf'table\s+{table_num + 1}[.:]\s*([^\n]+)',
                r'table\s+\d+[.:]\s*([^\n]+)',
            ]
            
            for pattern in caption_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            return ""
        except:
            return ""
    
    def _extract_figures(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract figure information from PDF"""
        figures = []
        
        try:
            text = self.extractor.extract_text(pdf_path)
            
            # Find figure references and captions
            figure_patterns = [
                r'figure\s+(\d+)[.:]\s*([^\n]+)',
                r'fig\.?\s+(\d+)[.:]\s*([^\n]+)',
            ]
            
            for pattern in figure_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    figure_data = {
                        'figure_number': int(match.group(1)),
                        'caption': match.group(2).strip(),
                        'position': match.span()
                    }
                    figures.append(figure_data)
        
        except Exception as e:
            logger.warning(f"Figure extraction failed: {str(e)}")
        
        return figures
    
    def _extract_references(self, full_text: str, references_section: str) -> List[str]:
        """Extract individual references"""
        references = []
        
        # Use references section if available
        ref_text = references_section if references_section else full_text
        
        # Split references by number
        ref_pattern = r'(?:^|\n)\s*(\d+)\.\s*([^\n]+(?:\n(?!\s*\d+\.)[^\n]+)*)'
        matches = re.findall(ref_pattern, ref_text, re.MULTILINE)
        
        for number, ref_content in matches:
            # Clean up reference text
            ref_content = re.sub(r'\s+', ' ', ref_content.strip())
            references.append(ref_content)
        
        return references
    
    def _extract_page_texts(self, pdf_path: str) -> List[str]:
        """Extract text from each page separately"""
        page_texts = []
        
        try:
            if PDFPLUMBER_AVAILABLE:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        page_texts.append(page_text or "")
            else:
                # Fallback using other methods
                full_text = self.extractor.extract_text(pdf_path)
                # Simple page splitting (not very accurate)
                estimated_pages = max(1, len(full_text) // 2000)  # Rough estimate
                page_size = len(full_text) // estimated_pages
                
                for i in range(estimated_pages):
                    start = i * page_size
                    end = (i + 1) * page_size if i < estimated_pages - 1 else len(full_text)
                    page_texts.append(full_text[start:end])
        
        except Exception as e:
            logger.warning(f"Page text extraction failed: {str(e)}")
        
        return page_texts
    
    def _create_empty_content(self, metadata: PDFMetadata) -> ExtractedContent:
        """Create empty ExtractedContent object"""
        return ExtractedContent(
            text="",
            metadata=metadata,
            sections={},
            tables=[],
            figures=[],
            references=[],
            page_texts=[]
        )
    
    def extract_clinical_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract clinical data specific patterns from medical PDF"""
        content = self.extract_structured_content(pdf_path)
        
        clinical_data = {
            'study_type': self._identify_study_type(content.text),
            'sample_size': self._extract_sample_size(content.text),
            'demographics': self._extract_demographics(content.text),
            'interventions': self._extract_interventions(content.text),
            'outcomes': self._extract_outcomes(content.text),
            'statistical_measures': self._extract_statistical_measures(content.text),
            'adverse_events': self._extract_adverse_events(content.text)
        }
        
        return clinical_data
    
    def _identify_study_type(self, text: str) -> str:
        """Identify type of clinical study"""
        study_types = {
            'randomized controlled trial': r'randomized\s+controlled\s+trial|RCT',
            'cohort study': r'cohort\s+study|prospective\s+study',
            'case-control study': r'case-control\s+study',
            'cross-sectional study': r'cross-sectional\s+study',
            'systematic review': r'systematic\s+review',
            'meta-analysis': r'meta-analysis',
            'case report': r'case\s+report',
            'case series': r'case\s+series'
        }
        
        for study_type, pattern in study_types.items():
            if re.search(pattern, text, re.IGNORECASE):
                return study_type
        
        return 'unknown'
    
    def _extract_sample_size(self, text: str) -> Dict[str, Any]:
        """Extract sample size information"""
        sample_patterns = [
            r'n\s*=\s*(\d+)',
            r'(\d+)\s+patients?',
            r'(\d+)\s+subjects?',
            r'(\d+)\s+participants?',
            r'sample\s+size\s+of\s+(\d+)'
        ]
        
        sample_sizes = []
        for pattern in sample_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sample_sizes.extend([int(n) for n in matches])
        
        return {
            'total': max(sample_sizes) if sample_sizes else None,
            'all_mentioned': list(set(sample_sizes))
        }
    
    def _extract_demographics(self, text: str) -> Dict[str, Any]:
        """Extract demographic information"""
        demographics = {}
        
        # Age patterns
        age_patterns = [
            r'mean\s+age[:\s]+(\d+\.?\d*)',
            r'age[:\s]+(\d+\.?\d*)\s*±\s*(\d+\.?\d*)',
            r'aged\s+(\d+)-(\d+)',
            r'(\d+)\s+years?\s+old'
        ]
        
        ages = []
        for pattern in age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    ages.extend([float(x) for x in match if x])
                else:
                    ages.append(float(match))
        
        if ages:
            demographics['age'] = {
                'mean': sum(ages) / len(ages),
                'values': ages
            }
        
        # Gender patterns
        gender_pattern = r'(\d+)\s*\(\s*(\d+\.?\d*)%\s*\)\s*(?:male|female)'
        gender_matches = re.findall(gender_pattern, text, re.IGNORECASE)
        
        if gender_matches:
            demographics['gender_distribution'] = gender_matches
        
        return demographics
    
    def _extract_interventions(self, text: str) -> List[str]:
        """Extract intervention descriptions"""
        intervention_patterns = [
            r'treatment\s+with\s+([^.]+)',
            r'administered\s+([^.]+)',
            r'received\s+([^.]+)',
            r'intervention[:\s]+([^.]+)'
        ]
        
        interventions = []
        for pattern in intervention_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            interventions.extend([match.strip() for match in matches])
        
        return list(set(interventions))[:10]  # Limit to 10 interventions
    
    def _extract_outcomes(self, text: str) -> Dict[str, Any]:
        """Extract outcome measures"""
        outcomes = {
            'primary_outcomes': [],
            'secondary_outcomes': [],
            'safety_outcomes': []
        }
        
        # Primary outcome patterns
        primary_patterns = [
            r'primary\s+(?:outcome|endpoint)[:\s]+([^.]+)',
            r'main\s+outcome[:\s]+([^.]+)'
        ]
        
        for pattern in primary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            outcomes['primary_outcomes'].extend(matches)
        
        # Secondary outcome patterns
        secondary_patterns = [
            r'secondary\s+(?:outcome|endpoint)[:\s]+([^.]+)',
        ]
        
        for pattern in secondary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            outcomes['secondary_outcomes'].extend(matches)
        
        return outcomes
    
    def _extract_statistical_measures(self, text: str) -> Dict[str, List[str]]:
        """Extract statistical measures and results"""
        measures = {
            'p_values': re.findall(r'p\s*[<>=]\s*0\.\d+', text, re.IGNORECASE),
            'confidence_intervals': re.findall(r'95%\s*CI[:\s]*\[?([^\]]+)\]?', text, re.IGNORECASE),
            'odds_ratios': re.findall(r'OR[:\s]*(\d+\.?\d*)', text, re.IGNORECASE),
            'hazard_ratios': re.findall(r'HR[:\s]*(\d+\.?\d*)', text, re.IGNORECASE),
            'relative_risks': re.findall(r'RR[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
        }
        
        return measures
    
    def _extract_adverse_events(self, text: str) -> List[str]:
        """Extract adverse events information"""
        ae_patterns = [
            r'adverse\s+events?[:\s]+([^.]+)',
            r'side\s+effects?[:\s]+([^.]+)',
            r'complications?[:\s]+([^.]+)',
            r'safety\s+concerns?[:\s]+([^.]+)'
        ]
        
        adverse_events = []
        for pattern in ae_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            adverse_events.extend([match.strip() for match in matches])
        
        return adverse_events


class PDFStructureParser:
    """Parse structured elements from PDFs"""
    
    def __init__(self):
        """Initialize structure parser"""
        pass
    
    def parse_bibliography(self, references_text: str) -> List[Dict[str, Any]]:
        """Parse bibliography section into structured references"""
        references = []
        
        # Split by reference numbers
        ref_pattern = r'(?:^|\n)\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.\s*|\Z)'
        matches = re.findall(ref_pattern, references_text, re.DOTALL)
        
        for number, ref_text in matches:
            parsed_ref = self._parse_reference(ref_text.strip())
            parsed_ref['number'] = int(number)
            references.append(parsed_ref)
        
        return references
    
    def _parse_reference(self, ref_text: str) -> Dict[str, Any]:
        """Parse individual reference"""
        reference = {
            'authors': '',
            'title': '',
            'journal': '',
            'year': '',
            'volume': '',
            'issue': '',
            'pages': '',
            'doi': '',
            'pmid': '',
            'type': 'journal'
        }
        
        # Extract DOI
        doi_match = re.search(r'doi:\s*([^\s,]+)', ref_text, re.IGNORECASE)
        if doi_match:
            reference['doi'] = doi_match.group(1)
        
        # Extract PMID
        pmid_match = re.search(r'pmid:\s*(\d+)', ref_text, re.IGNORECASE)
        if pmid_match:
            reference['pmid'] = pmid_match.group(1)
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
        if year_match:
            reference['year'] = year_match.group(0)
        
        # Extract volume and pages
        vol_pages_match = re.search(r'(\d+)\s*(?:\((\d+)\))?\s*:\s*(\d+)-?(\d+)?', ref_text)
        if vol_pages_match:
            reference['volume'] = vol_pages_match.group(1)
            if vol_pages_match.group(2):
                reference['issue'] = vol_pages_match.group(2)
            pages = vol_pages_match.group(3)
            if vol_pages_match.group(4):
                pages += f"-{vol_pages_match.group(4)}"
            reference['pages'] = pages
        
        return reference


# Convenience functions
def extract_pdf_text(pdf_path: str, method: str = 'auto') -> str:
    """
    Convenience function to extract text from PDF
    
    Args:
        pdf_path: Path to PDF file
        method: Extraction method
        
    Returns:
        Extracted text
    """
    extractor = PDFExtractor()
    return extractor.extract_text(pdf_path, method)


def extract_pdf_metadata(pdf_path: str) -> PDFMetadata:
    """
    Convenience function to extract PDF metadata
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        PDFMetadata object
    """
    extractor = PDFExtractor()
    return extractor.extract_metadata(pdf_path)


def process_medical_papers(pdf_directory: str,
                          output_directory: str = None) -> List[ExtractedContent]:
    """
    Process multiple medical papers from directory
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Optional output directory for processed files
        
    Returns:
        List of ExtractedContent objects
    """
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        logger.error(f"Directory not found: {pdf_directory}")
        return []
    
    processor = MedicalPDFProcessor()
    extracted_contents = []
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        
        try:
            content = processor.extract_structured_content(str(pdf_file))
            extracted_contents.append(content)
            
            # Save to output directory if specified
            if output_directory:
                output_path = Path(output_directory)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save as JSON
                output_file = output_path / f"{pdf_file.stem}.json"
                content_dict = {
                    'text': content.text,
                    'sections': content.sections,
                    'tables': content.tables,
                    'figures': content.figures,
                    'references': content.references,
                    'metadata': {
                        'title': content.metadata.title,
                        'author': content.metadata.author,
                        'pages': content.metadata.pages,
                        'doi': content.metadata.doi,
                        'journal': content.metadata.journal
                    }
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(content_dict, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
    
    logger.info(f"Successfully processed {len(extracted_contents)} files")
    return extracted_contents


def batch_process_pdfs(pdf_paths: List[str],
                      output_format: str = 'json') -> Dict[str, Any]:
    """
    Batch process multiple PDFs
    
    Args:
        pdf_paths: List of PDF file paths
        output_format: Output format ('json', 'text', 'structured')
        
    Returns:
        Dictionary with processing results
    """
    processor = MedicalPDFProcessor()
    results = {
        'successful': [],
        'failed': [],
        'total_files': len(pdf_paths),
        'extraction_stats': {}
    }
    
    for pdf_path in pdf_paths:
        try:
            if output_format == 'structured':
                content = processor.extract_structured_content(pdf_path)
                results['successful'].append({
                    'file': pdf_path,
                    'content': content
                })
            elif output_format == 'text':
                text = extract_pdf_text(pdf_path)
                results['successful'].append({
                    'file': pdf_path,
                    'text': text
                })
            else:  # json
                content = processor.extract_structured_content(pdf_path)
                results['successful'].append({
                    'file': pdf_path,
                    'data': {
                        'text': content.text,
                        'sections': content.sections,
                        'tables': len(content.tables),
                        'figures': len(content.figures),
                        'references': len(content.references)
                    }
                })
        
        except Exception as e:
            results['failed'].append({
                'file': pdf_path,
                'error': str(e)
            })
    
    # Calculate statistics
    results['extraction_stats'] = {
        'success_rate': len(results['successful']) / len(pdf_paths),
        'failed_count': len(results['failed']),
        'successful_count': len(results['successful'])
    }
    
    return results


def validate_pdf_structure(pdf_path: str) -> Dict[str, Any]:
    """
    Validate PDF structure and content quality
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Validation results
    """
    validation = {
        'is_valid': False,
        'has_text': False,
        'has_metadata': False,
        'estimated_quality': 'unknown',
        'issues': [],
        'recommendations': []
    }
    
    try:
        extractor = PDFExtractor()
        
        # Basic file validation
        if not Path(pdf_path).exists():
            validation['issues'].append('File does not exist')
            return validation
        
        # Extract text and metadata
        text = extractor.extract_text(pdf_path)
        metadata = extractor.extract_metadata(pdf_path)
        
        validation['is_valid'] = True
        
        # Check text content
        if text and len(text.strip()) > 100:
            validation['has_text'] = True
        else:
            validation['issues'].append('Minimal or no text content')
            validation['recommendations'].append('Consider using OCR extraction')
        
        # Check metadata
        if metadata.title or metadata.author:
            validation['has_metadata'] = True
        else:
            validation['issues'].append('Limited metadata available')
        
        # Estimate quality
        if validation['has_text'] and validation['has_metadata']:
            validation['estimated_quality'] = 'good'
        elif validation['has_text']:
            validation['estimated_quality'] = 'fair'
        else:
            validation['estimated_quality'] = 'poor'
            validation['recommendations'].append('File may be scanned/image-based')
        
        # Check for medical content
        medical_keywords = ['patient', 'treatment', 'diagnosis', 'clinical', 'medical', 'study']
        if any(keyword in text.lower() for keyword in medical_keywords):
            validation['appears_medical'] = True
        else:
            validation['appears_medical'] = False
            validation['recommendations'].append('Content may not be medical literature')
    
    except Exception as e:
        validation['issues'].append(f'Validation error: {str(e)}')
    
    return validation


# Export all classes and functions
__all__ = [
    'PDFExtractor',
    'MedicalPDFProcessor',
    'PDFStructureParser',
    'PDFMetadata',
    'ExtractedContent',
    'extract_pdf_text',
    'extract_pdf_metadata',
    'process_medical_papers',
    'batch_process_pdfs',
    'validate_pdf_structure'
]