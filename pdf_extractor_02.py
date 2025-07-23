#!/usr/bin/env python3
"""
Enterprise-grade PDF Data Extractor for Technical Equipment Tag Recognition
Author: Enterprise Development Team
Version: 1.0.0
Python: 3.8+
"""

import re
import csv
import gc
import os
import time
import psutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Protocol, Union
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymupdf  # PyMuPDF - highest performance PDF library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)

class PDFExtractionError(Exception):
    """Base exception for PDF extraction operations."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "PDF_EXTRACTION_ERROR"
        self.context = context or {}
        super().__init__(self.message)

class PDFFileNotFoundError(PDFExtractionError):
    """Raised when PDF file is not found."""
    
    def __init__(self, file_path: Path):
        super().__init__(
            f"PDF file not found: {file_path}",
            error_code="PDF_FILE_NOT_FOUND",
            context={"file_path": str(file_path)}
        )

class PDFCorruptedError(PDFExtractionError):
    """Raised when PDF file is corrupted or unreadable."""
    pass

class PDFExtractionTimeoutError(PDFExtractionError):
    """Raised when extraction exceeds timeout limit."""
    pass

@dataclass(frozen=True)
class ExtractionResult:
    """Immutable data class for extraction results."""
    document_path: Path
    extracted_data: List[Dict[str, Any]]
    processing_metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0

@dataclass
class ProcessingMetrics:
    """Metrics collector for performance monitoring."""
    file_path: str
    file_size_mb: float
    page_count: int
    processing_time_seconds: float
    peak_memory_mb: float
    extracted_tags_count: int
    error_count: int
    start_time: datetime = field(default_factory=datetime.now)

class RegexPatternManager:
    """Manages and validates regex patterns for equipment identification."""
    
    # Optimized regex patterns based on research findings
    EQUIPMENT_PATTERNS = {
        'bomba': {
            'pattern': re.compile(r'\b\d{3}P\d{4}(?:[A-Z](?:/[A-Z])?)?\b'),
            'description': 'Pump equipment (optimized from original pattern)'
        },
        'horno': {
            'pattern': re.compile(r'\b\d{3}F\d{4}(?:[A-Z](?:/[A-Z])?)?\b'),
            'description': 'Furnace equipment (optimized from original pattern)'
        },
        'intercambiador': {
            'pattern': re.compile(r'\b\d{3}E\d{4}[A-Z]?\b'),
            'description': 'Heat exchanger equipment'
        },
        'recipiente': {
            'pattern': re.compile(r'\b\d{3}V\d{4}[A-Z]?\b'),
            'description': 'Vessel equipment'
        },
        'compresor': {
            'pattern': re.compile(r'\b\d{3}C\d{4}[A-Z]?\b'),
            'description': 'Compressor equipment'
        },
        'instrumento': {
            'pattern': re.compile(r'\b[A-Z]{2,4}-\d{3,5}[A-Z]?\b'),
            'description': 'Instrument tags (ISA standard)'
        },
        'tuberia': {
            'pattern': re.compile(r'\b\d{1,2}"-[A-Z]{2,4}-\d{4,6}-[A-Z0-9-]{1,20}\b'),
            'description': 'Piping system (optimized to prevent backtracking)'
        }
    }
    
    # @classmethod
    # def validate_patterns(cls) -> Dict[str, bool]:
        # """Validate all regex patterns."""
        # validation_results = {}
        
        # for pattern_name, pattern_info in cls.EQUIPMENT_PATTERNS.items():
            # try:
                # # Test compilation
                # pattern = pattern_info['pattern']
                
                # # Test with sample strings
                # test_strings = cls.get_test_strings(pattern_name)
                # matches = any(pattern.search(test_str) for test_str in test_strings)
                
                # validation_results[pattern_name] = matches
                
            # except re.error as e:
                # logging.error(f"Invalid regex pattern for {pattern_name}: {e}")
                # validation_results[pattern_name] = False
        
        # return validation_results
    
    @classmethod
    def get_test_strings(cls, pattern_name: str) -> List[str]:
        """Get test strings for pattern validation."""
        test_data = {
            'bomba': ['123P4567', '456P7890A', '789P0123B/C', '001P2345X'],
            'horno': ['100F2500', '205F3678A', '304F5555B/D', '999F1111X/Z'],
            'intercambiador': ['123E4567', '456E7890A', '001E2345Z', '999E8888B'],
            'recipiente': ['123V4567', '789V0123A', '555V9999Z'],
            'compresor': ['123C4567', '456C7890B', '999C1111A'],
            'instrumento': ['PT-123', 'FIC-12345A', 'TRC-4567B', 'LAHH-999', 'AI-12345'],
            'tuberia': ['6"-PT-1234-STEAM', '12"-FW-123456-HOT-WATER', '8"-CW-5678-COOLING']
        }
        return test_data.get(pattern_name, [])

class MemoryMonitor:
    """Memory usage monitoring and management."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.peak_memory = 0
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def check_memory_threshold(self) -> bool:
        """Check if memory usage exceeds threshold."""
        current_memory = self.get_current_memory_mb()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if current_memory > self.max_memory_mb:
            gc.collect()  # Force garbage collection
            current_memory = self.get_current_memory_mb()
            
            if current_memory > self.max_memory_mb:
                raise MemoryError(f"Memory usage {current_memory:.1f}MB exceeds limit {self.max_memory_mb}MB")
        
        return current_memory > self.max_memory_mb * 0.8  # Warning threshold

class PDFReader:
    """Optimized PDF reading implementation using PyMuPDF."""
    
    def __init__(self, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def open_pdf(self, pdf_path: Path):
        """Context manager for PDF document handling."""
        try:
            doc = pymupdf.open(str(pdf_path))
            
            if doc.needs_pass:
                doc.close()
                raise PDFExtractionError(
                    f"PDF is password protected: {pdf_path}",
                    error_code="PASSWORD_PROTECTED"
                )
            
            yield doc
            
        except pymupdf.FileNotFoundError:
            raise PDFFileNotFoundError(pdf_path)
        except pymupdf.FileDataError:
            raise PDFCorruptedError(f"PDF file appears to be corrupted: {pdf_path}")
        finally:
            if 'doc' in locals():
                doc.close()
            gc.collect()
    
    def extract_text_chunked(self, pdf_path: Path) -> Iterator[tuple[str, int]]:
        """Extract text in chunks for memory efficiency."""
        with self.open_pdf(pdf_path) as doc:
            total_pages = doc.page_count
            
            for start_page in range(0, total_pages, self.chunk_size):
                end_page = min(start_page + self.chunk_size, total_pages)
                chunk_text_parts = []
                
                for page_num in range(start_page, end_page):
                    try:
                        page = doc[page_num]
                        text = page.get_text()
                        
                        if text.strip():  # Skip empty pages
                            chunk_text_parts.append((text, page_num + 1))
                    
                    except Exception as e:
                        self.logger.warning(f"Error reading page {page_num + 1}: {e}")
                        continue
                
                # Yield chunk results
                for text, page_num in chunk_text_parts:
                    yield text, page_num
                
                # Cleanup after each chunk
                del chunk_text_parts
                gc.collect()

class DataExtractor:
    """Core data extraction logic using regex patterns."""
    
    def __init__(self, pattern_manager: PatternManager):
        self.pattern_manager = pattern_manager
        self.logger = logging.getLogger(__name__)
    
    def extract_tags_from_text(self, text: str, page_number: int, document_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts, processes, and categorizes tags from a text chunk.
        It uses a dispatch method to handle different tag structures based on the matched pattern.
        """
        all_matches_details = self.pattern_manager.find_all_matches(text)
        extracted_data = []
        
        for match_details in all_matches_details:
            original_tag_text = match_details['tag_capturado']
            pattern_name = match_details['pattern_name']
            match_obj = match_details['match_obj']
            
            # --- Lógica de Despacho (Dispatcher) ---
            if pattern_name == "instrumento_con_unidad":
                tag_info = self._process_instrumento_con_unidad(match_obj)
            elif pattern_name == "unidad_tag_sin_guion":
                tag_info = self._process_equipo_con_unidad(match_obj)
            elif pattern_name == "instrumento_sin_unidad":
                tag_info = self._process_tag_simple(match_obj)
            elif pattern_name == "componente_electrico_comun":
                 tag_info = self._process_tag_simple(match_obj)
            else: # Fallback para otros patrones (documentos, tuberías, etc.)
                tag_info = self._process_tag_simple(match_obj)

            # Enriquecer con metadatos comunes
            tag_info.update({
                'documento_origen': str(document_path.name),
                'tipo_elemento': self.pattern_manager.patterns[pattern_name]['category'],
                'pagina_encontrada': page_number,
                'contexto_linea': match_details['context'],
                'posicion_inicio': match_details['start'],
                'posicion_fin': match_details['end'],
            })
            
            extracted_data.append(tag_info)
            
        return extracted_data

    # --- MÉTODOS DE PROCESAMIENTO PRIVADOS ---

    def _process_tag_simple(self, match_obj: re.Match) -> Dict[str, Any]:
        """Procesa un tag simple sin unidad extraíble."""
        tag = match_obj.group(1)
        tag_base = tag[:-1] if tag[-1] in 'ABCD' else tag
        sufijo = tag[-1] if tag[-1] in 'ABCD' else None
        return {'tag_capturado': tag, 'tag_normalizado': tag, 'unidad_extraida': None, 'tag_base': tag_base, 'sufijo_equipo': sufijo}

    def _process_instrumento_con_unidad(self, match_obj: re.Match) -> Dict[str, Any]:
        """Procesa tags como '010-AI-0002'."""
        unidad = match_obj.group(1).replace('-', '')
        tag_normalizado = match_obj.group(2)
        tag_base = tag_normalizado[:-1] if tag_normalizado[-1] in 'ABCD' else tag_normalizado
        sufijo = tag_normalizado[-1] if tag_normalizado[-1] in 'ABCD' else None
        return {'tag_capturado': match_obj.group(0), 'tag_normalizado': tag_normalizado, 'unidad_extraida': unidad, 'tag_base': tag_base, 'sufijo_equipo': sufijo}

    def _process_equipo_con_unidad(self, match_obj: re.Match) -> Dict[str, Any]:
        """Procesa tags como '011C0001'."""
        unidad = match_obj.group(1)
        tipo = match_obj.group(2)
        numero_con_sufijo = match_obj.group(3)
        tag_normalizado = f"{tipo}{numero_con_sufijo}"
        tag_base = f"{tipo}{numero_con_sufijo[:-1]}" if numero_con_sufijo[-1] in 'ABCD' else tag_normalizado
        sufijo = numero_con_sufijo[-1] if numero_con_sufijo[-1] in 'ABCD' else None
        return {'tag_capturado': match_obj.group(0), 'tag_normalizado': tag_normalizado, 'unidad_extraida': unidad, 'tag_base': tag_base, 'sufijo_equipo': sufijo}
        
class CSVWriter:
    """Optimized CSV output handling."""
    
    def __init__(self, output_directory: Path = Path("output")):
        self.output_directory = output_directory
        self.output_directory.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def write_results(self, results: List[Dict[str, Any]], output_filename: str) -> Path:
        """Write extraction results to CSV file."""
        output_path = self.output_directory / output_filename
        
        if not results:
            self.logger.warning("No data to write to CSV")
            return output_path
        
        fieldnames = [
            'documento_origen',
            'tag_capturado', 
            'tipo_elemento',
            'pagina_encontrada',
            'contexto_linea',
            'posicion_inicio',
            'posicion_fin'
        ]
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write in batches for large datasets
                batch_size = 1000
                for i in range(0, len(results), batch_size):
                    batch = results[i:i + batch_size]
                    for row in batch:
                        # Clean row data
                        cleaned_row = {k: str(v) if v is not None else '' for k, v in row.items()}
                        writer.writerow(cleaned_row)
                    
                    csvfile.flush()  # Force write to disk
            
            self.logger.info(f"Successfully wrote {len(results)} records to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error writing CSV file: {e}")
            raise

class PDFDataExtractor:
    """
    Enterprise-grade PDF Data Extractor for technical equipment tag recognition.
    
    Features:
    - Memory-efficient chunk processing for large PDFs (10MB-500MB)
    - Optimized regex patterns for equipment identification
    - Comprehensive error handling and logging
    - Performance metrics collection
    - Multi-processing support for batch operations
    """
    
    def __init__(
        self,
        chunk_size: int = 10,
        max_memory_mb: int = 1024,
        extraction_timeout: int = 300,
        output_directory: str = "output",
        pattern_manager: Optional[RegexPatternManager] = None
    ):
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.extraction_timeout = extraction_timeout
        
        # Initialize components
        #self.pattern_manager = RegexPatternManager()
        self.pattern_manager = pattern_manager if pattern_manager is not None else RegexPatternManager()
        self.pdf_reader = PDFReader(chunk_size=chunk_size)
        self.data_extractor = DataExtractor(self.pattern_manager)
        self.csv_writer = CSVWriter(Path(output_directory))
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # # Validate patterns on initialization
        # validation_results = self.pattern_manager.validate_patterns()
        # self.logger.info(f"Pattern validation results: {validation_results}")
        
        # if not all(validation_results.values()):
            # failed_patterns = [k for k, v in validation_results.items() if not v]
            # self.logger.warning(f"Some patterns failed validation: {failed_patterns}")
    
    def extract_from_single_pdf(self, pdf_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract data from a single PDF file with comprehensive error handling.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractionResult: Comprehensive extraction results with metrics
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()
        
        # Initialize metrics
        file_size_mb = pdf_path.stat().st_size / 1024 / 1024
        metrics = ProcessingMetrics(
            file_path=str(pdf_path),
            file_size_mb=file_size_mb,
            page_count=0,
            processing_time_seconds=0.0,
            peak_memory_mb=0.0,
            extracted_tags_count=0,
            error_count=0
        )
        
        try:
            self.logger.info(f"Starting extraction for: {pdf_path} ({file_size_mb:.2f} MB)")
            
            # Validate file existence and size
            if not pdf_path.exists():
                raise PDFFileNotFoundError(pdf_path)
            
            if file_size_mb > 500:  # 500MB limit
                raise PDFExtractionError(
                    f"File size {file_size_mb:.2f}MB exceeds maximum limit of 500MB",
                    error_code="FILE_TOO_LARGE"
                )
            
            # Extract data with memory monitoring
            all_extracted_data = []
            
            for text_chunk, page_number in self.pdf_reader.extract_text_chunked(pdf_path):
                metrics.page_count = page_number  # Update page count
                
                # Check memory usage
                if self.memory_monitor.check_memory_threshold():
                    self.logger.warning(f"Memory usage approaching limit during page {page_number}")
                
                # Extract tags from current chunk
                chunk_data = self.data_extractor.extract_tags_from_text(
                    text_chunk, page_number, pdf_path
                )
                all_extracted_data.extend(chunk_data)
                
                # Update metrics
                metrics.extracted_tags_count = len(all_extracted_data)
                metrics.peak_memory_mb = self.memory_monitor.peak_memory
            
            # Calculate final metrics
            processing_time = time.time() - start_time
            metrics.processing_time_seconds = processing_time
            
            self.logger.info(
                f"Extraction completed: {pdf_path} | "
                f"Pages: {metrics.page_count} | "
                f"Tags: {metrics.extracted_tags_count} | "
                f"Time: {processing_time:.2f}s | "
                f"Peak Memory: {metrics.peak_memory_mb:.1f}MB"
            )
            
            return ExtractionResult(
                document_path=pdf_path,
                extracted_data=all_extracted_data,
                processing_metrics=metrics.__dict__,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            metrics.error_count = 1
            metrics.processing_time_seconds = processing_time
            
            self.logger.error(f"Extraction failed for {pdf_path}: {e}")
            
            return ExtractionResult(
                document_path=pdf_path,
                extracted_data=[],
                processing_metrics=metrics.__dict__,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def extract_from_multiple_pdfs(
        self, 
        pdf_paths: List[Union[str, Path]], 
        max_workers: int = None
    ) -> List[ExtractionResult]:
        """
        Extract data from multiple PDF files using multiprocessing.
        
        Args:
            pdf_paths: List of PDF file paths
            max_workers: Maximum number of worker processes
            
        Returns:
            List[ExtractionResult]: Results for all processed files
        """
        max_workers = max_workers or min(4, len(pdf_paths))
        results = []
        
        self.logger.info(f"Starting batch processing of {len(pdf_paths)} files with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_pdf_worker, str(pdf_path)): pdf_path 
                for pdf_path in pdf_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        self.logger.info(f"Completed: {pdf_path}")
                    else:
                        self.logger.error(f"Failed: {pdf_path} - {result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"Worker process failed for {pdf_path}: {e}")
                    results.append(ExtractionResult(
                        document_path=Path(pdf_path),
                        extracted_data=[],
                        processing_metrics={},
                        success=False,
                        error_message=str(e)
                    ))
        
        return results
    
    @staticmethod
    def _process_single_pdf_worker(pdf_path: str) -> ExtractionResult:
        """Worker function for multiprocessing (must be static)."""
        # Create new extractor instance for this worker
        extractor = PDFDataExtractor()
        return extractor.extract_from_single_pdf(pdf_path)
    
    def save_results_to_csv(
        self, 
        results: List[ExtractionResult], 
        output_filename: str = None
    ) -> Path:
        """
        Save extraction results to CSV file.
        
        Args:
            results: List of extraction results
            output_filename: Output CSV filename (auto-generated if None)
            
        Returns:
            Path: Path to the created CSV file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"extracted_data_{timestamp}.csv"
        
        # Combine all extracted data
        all_data = []
        for result in results:
            if result.success and result.extracted_data:
                all_data.extend(result.extracted_data)
        
        if not all_data:
            self.logger.warning("No data to save to CSV")
            return self.csv_writer.output_directory / output_filename
        
        return self.csv_writer.write_results(all_data, output_filename)
    
    def get_performance_summary(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        """Generate performance summary from extraction results."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results:
            return {
                'total_files': len(results),
                'successful_files': 0,
                'failed_files': len(failed_results),
                'success_rate': 0.0
            }
        
        processing_times = [r.processing_time for r in successful_results]
        total_tags = sum(len(r.extracted_data) for r in successful_results)
        
        return {
            'total_files': len(results),
            'successful_files': len(successful_results),
            'failed_files': len(failed_results),
            'success_rate': len(successful_results) / len(results),
            'total_tags_extracted': total_tags,
            'avg_processing_time': sum(processing_times) / len(processing_times),
            'max_processing_time': max(processing_times),
            'min_processing_time': min(processing_times),
            'throughput_files_per_second': len(successful_results) / sum(processing_times) if sum(processing_times) > 0 else 0
        }

# Example usage and testing
def main():
    """Example usage of PDFDataExtractor."""
    # Initialize extractor
    extractor = PDFDataExtractor(
        chunk_size=10,
        max_memory_mb=512,
        extraction_timeout=300,
        output_directory="output"
    )
    
    # Example: Process single PDF
    try:
        pdf_path = Path("example.pdf")  # Replace with actual PDF path
        result = extractor.extract_from_single_pdf(pdf_path)
        
        if result.success:
            print(f"Successfully extracted {len(result.extracted_data)} tags")
            
            # Save to CSV
            csv_path = extractor.save_results_to_csv([result])
            print(f"Results saved to: {csv_path}")
        else:
            print(f"Extraction failed: {result.error_message}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()