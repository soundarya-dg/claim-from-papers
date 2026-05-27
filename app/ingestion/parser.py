import fitz
import re
from pathlib import Path
from typing import List, Dict, Optional


class PDFParser:
    """Parses PDF files and extracts text with metadata."""
    
    def __init__(self):
        pass
    
    def extract_title_from_filename(self, filename: str) -> str:

        # Remove file extension
        title = Path(filename).stem
        # Remove arXiv ID prefix if present
        title = re.sub(r'^\d+\.\d+v\d+_', '', title)
        # Replace underscores with spaces
        title = title.replace('_', ' ')
        # Clean up multiple spaces
        title = re.sub(r'\s+', ' ', title).strip()
        return title
    
    def parse_pdf(self, pdf_path: str, extract_title: bool = True) -> Dict[str, any]:
        """
        Parse a PDF file and extract text with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            extract_title: Whether to extract title from filename
            
        Returns: Dict containing: title, pages, total_pages, filename
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract title from filename
        title = self.extract_title_from_filename(pdf_path.name) if extract_title else pdf_path.stem
        
        # Open the PDF
        try:
            doc = fitz.open(pdf_path)
            
            # Extract text from each page
            pages = []
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean up text
                text = self._clean_text(text)
                
                # Only add pages with actual content
                if text.strip():
                    pages.append({
                        'page_number': page_num + 1,  # 1-indexed
                        'text': text,
                        'metadata': {
                            'title': title,
                            'page': page_num + 1,
                            'filename': pdf_path.name,
                            'total_pages': total_pages
                        }
                    })
            
            # Close document after processing all pages
            doc.close()
            
            return {
                'title': title,
                'pages': pages,
                'total_pages': total_pages,
                'filename': pdf_path.name
            }
            
        except Exception as e:
            raise ValueError(f"Failed to process PDF {pdf_path}: {e}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove page headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip very short lines at the beginning/end - often headers/footers
            if len(line.strip()) > 0:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Normalize whitespace
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def parse_directory(self, directory_path: str, file_pattern: str = "*.pdf") -> List[Dict[str, any]]:
        """
        Parse all PDFs in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            file_pattern: Glob pattern for PDF files
            
        Returns:
            List of parsed document dictionaries
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        # Find all PDF files
        pdf_files = sorted(directory.glob(file_pattern))
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {directory}")
            return []
        
        print(f"\nParsing {len(pdf_files)} PDF files from {directory}...\n")
        
        parsed_docs = []
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                print(f"[{i}/{len(pdf_files)}] Parsing: {pdf_file.name}")
                parsed_doc = self.parse_pdf(pdf_file)
                parsed_docs.append(parsed_doc)
                print(f"    Extracted {len(parsed_doc['pages'])} pages")
            except Exception as e:
                print(f"    Error parsing {pdf_file.name}: {e}")
                continue
        
        print(f"\n Successfully parsed {len(parsed_docs)} documents")
        return parsed_docs