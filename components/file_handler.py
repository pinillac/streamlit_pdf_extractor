"""
File Handler Component
Manages file uploads, validation, and pre-processing for the PDF extractor
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
import hashlib
from typing import List, Dict, Any, Optional, Union
import PyPDF2
import os
import magic
from datetime import datetime

class FileHandler:
    """Handles file operations including upload, validation, and temporary storage"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "pdf_extractor_temp"
        self.temp_dir.mkdir(exist_ok=True)
        self.max_file_size_mb = 500
        self.supported_types = ['.pdf']
        
    def validate_file(self, file) -> Dict[str, Any]:
        """
        Validate uploaded file for size, type, and integrity
        
        Args:
            file: Streamlit uploaded file object
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        # Check file size
        file_size_mb = file.size / (1024 * 1024)
        validation_result['file_info']['size_mb'] = file_size_mb
        
        if file_size_mb > self.max_file_size_mb:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({self.max_file_size_mb}MB)"
            )
        elif file_size_mb > 100:
            validation_result['warnings'].append(
                f"Large file detected ({file_size_mb:.1f}MB). Processing may take longer."
            )
        
        # Check file extension
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in self.supported_types:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Unsupported file type: {file_ext}. Only PDF files are supported."
            )
        
        # Validate PDF structure
        try:
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                validation_result['valid'] = False
                validation_result['errors'].append("PDF is encrypted/password protected")
            
            # Get page count
            page_count = len(pdf_reader.pages)
            validation_result['file_info']['page_count'] = page_count
            
            if page_count == 0:
                validation_result['valid'] = False
                validation_result['errors'].append("PDF has no pages")
            elif page_count > 1000:
                validation_result['warnings'].append(
                    f"PDF has {page_count} pages. Consider splitting for better performance."
                )
            
            # Check if text can be extracted
            file.seek(0)
            sample_text = ""
            for i in range(min(3, page_count)):  # Check first 3 pages
                try:
                    sample_text += pdf_reader.pages[i].extract_text()
                except:
                    pass
            
            if not sample_text.strip():
                validation_result['warnings'].append(
                    "No text found in sample pages. PDF might be scanned/image-based."
                )
            
            validation_result['file_info']['has_text'] = bool(sample_text.strip())
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"PDF validation error: {str(e)}")
        
        finally:
            file.seek(0)  # Reset file pointer
        
        return validation_result
    
    def save_uploaded_file(self, uploaded_file) -> Optional[Path]:
        """
        Save uploaded file to temporary directory
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Path to saved file or None if error
        """
        try:
            # Generate unique filename
            file_hash = hashlib.md5(uploaded_file.read()).hexdigest()[:8]
            uploaded_file.seek(0)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{Path(uploaded_file.name).stem}_{timestamp}_{file_hash}.pdf"
            
            file_path = self.temp_dir / filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            uploaded_file.seek(0)  # Reset for future use
            
            return file_path
            
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return None
    
    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract detailed metadata from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing file metadata
        """
        metadata = {
            'filename': file_path.name,
            'size_mb': file_path.stat().st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(file_path.stat().st_ctime),
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
        }
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                metadata['page_count'] = len(pdf_reader.pages)
                metadata['is_encrypted'] = pdf_reader.is_encrypted
                
                # Extract PDF metadata
                if pdf_reader.metadata:
                    metadata['pdf_metadata'] = {
                        'title': pdf_reader.metadata.get('/Title', 'N/A'),
                        'author': pdf_reader.metadata.get('/Author', 'N/A'),
                        'subject': pdf_reader.metadata.get('/Subject', 'N/A'),
                        'creator': pdf_reader.metadata.get('/Creator', 'N/A'),
                        'producer': pdf_reader.metadata.get('/Producer', 'N/A'),
                        'creation_date': str(pdf_reader.metadata.get('/CreationDate', 'N/A')),
                        'modification_date': str(pdf_reader.metadata.get('/ModDate', 'N/A'))
                    }
                
                # Sample text extraction for preview
                sample_pages = min(3, len(pdf_reader.pages))
                sample_text = ""
                for i in range(sample_pages):
                    try:
                        page_text = pdf_reader.pages[i].extract_text()
                        sample_text += page_text[:500] + "\n\n"
                    except:
                        pass
                
                metadata['sample_text'] = sample_text.strip()
                metadata['has_extractable_text'] = bool(sample_text.strip())
                
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def estimate_processing_time(self, files: List[Any]) -> Dict[str, Any]:
        """
        Estimate processing time based on file characteristics
        
        Args:
            files: List of file objects or paths
            
        Returns:
            Dictionary with time estimates
        """
        total_pages = 0
        total_size_mb = 0
        
        for file in files:
            if hasattr(file, 'size'):  # Streamlit uploaded file
                total_size_mb += file.size / (1024 * 1024)
                
                # Estimate pages based on file size (rough approximation)
                estimated_pages = int(file.size / (1024 * 50))  # ~50KB per page average
                total_pages += max(estimated_pages, 10)
            else:  # Path object
                if Path(file).exists():
                    total_size_mb += Path(file).stat().st_size / (1024 * 1024)
                    
                    try:
                        with open(file, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            total_pages += len(pdf_reader.pages)
                    except:
                        total_pages += 10  # Default estimate
        
        # Estimation formula (based on benchmarks)
        # ~1 page per second for text extraction + pattern matching
        base_time = total_pages * 1.0
        
        # Add overhead for large files
        if total_size_mb > 100:
            base_time *= 1.5
        
        # Add initialization time
        setup_time = 5
        
        # Calculate estimates
        estimated_time = base_time + setup_time
        
        return {
            'total_pages': total_pages,
            'total_size_mb': round(total_size_mb, 2),
            'estimated_seconds': round(estimated_time, 0),
            'estimated_minutes': round(estimated_time / 60, 1),
            'confidence': 'high' if total_pages < 1000 else 'medium'
        }
    
    def prepare_batch_processing(self, files: List[Any]) -> List[Dict[str, Any]]:
        """
        Prepare files for batch processing
        
        Args:
            files: List of uploaded files
            
        Returns:
            List of file information dictionaries
        """
        batch_info = []
        
        for idx, file in enumerate(files):
            # Validate file
            validation = self.validate_file(file)
            
            if validation['valid']:
                # Save to temp directory
                file_path = self.save_uploaded_file(file)
                
                if file_path:
                    batch_info.append({
                        'index': idx,
                        'original_name': file.name,
                        'temp_path': file_path,
                        'size_mb': validation['file_info']['size_mb'],
                        'page_count': validation['file_info'].get('page_count', 0),
                        'has_text': validation['file_info'].get('has_text', True),
                        'status': 'ready'
                    })
                else:
                    batch_info.append({
                        'index': idx,
                        'original_name': file.name,
                        'status': 'error',
                        'error': 'Failed to save file'
                    })
            else:
                batch_info.append({
                    'index': idx,
                    'original_name': file.name,
                    'status': 'invalid',
                    'errors': validation['errors']
                })
        
        return batch_info
    
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """
        Clean up temporary files older than specified hours
        
        Args:
            older_than_hours: Remove files older than this many hours
        """
        try:
            current_time = datetime.now()
            
            for file_path in self.temp_dir.glob("*.pdf"):
                file_age_hours = (current_time - datetime.fromtimestamp(
                    file_path.stat().st_mtime
                )).total_seconds() / 3600
                
                if file_age_hours > older_than_hours:
                    file_path.unlink()
                    
        except Exception as e:
            st.warning(f"Error during cleanup: {str(e)}")
    
    def get_mime_type(self, file_path: Path) -> str:
        """
        Get MIME type of file for validation
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME type string
        """
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(str(file_path))
        except:
            # Fallback to extension-based detection
            return 'application/pdf' if file_path.suffix.lower() == '.pdf' else 'unknown'
    
    def split_large_pdf(self, file_path: Path, max_pages_per_chunk: int = 100) -> List[Path]:
        """
        Split large PDF into smaller chunks for processing
        
        Args:
            file_path: Path to large PDF file
            max_pages_per_chunk: Maximum pages per chunk
            
        Returns:
            List of paths to chunk files
        """
        chunk_paths = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
                
                if total_pages <= max_pages_per_chunk:
                    return [file_path]  # No need to split
                
                # Create chunks
                for chunk_idx in range(0, total_pages, max_pages_per_chunk):
                    pdf_writer = PyPDF2.PdfWriter()
                    
                    # Add pages to chunk
                    end_page = min(chunk_idx + max_pages_per_chunk, total_pages)
                    for page_idx in range(chunk_idx, end_page):
                        pdf_writer.add_page(pdf_reader.pages[page_idx])
                    
                    # Save chunk
                    chunk_filename = f"{file_path.stem}_chunk_{chunk_idx//max_pages_per_chunk + 1}.pdf"
                    chunk_path = self.temp_dir / chunk_filename
                    
                    with open(chunk_path, 'wb') as chunk_file:
                        pdf_writer.write(chunk_file)
                    
                    chunk_paths.append(chunk_path)
                    
        except Exception as e:
            st.error(f"Error splitting PDF: {str(e)}")
            return [file_path]  # Return original file on error
        
        return chunk_paths