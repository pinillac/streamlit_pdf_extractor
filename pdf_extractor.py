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
import sqlite3
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
    
    def __init__(self, pattern_manager: RegexPatternManager):
        self.pattern_manager = pattern_manager
        self.logger = logging.getLogger(__name__)
    
    def _disaggregate_and_format_tag(self, tag: str) -> List[str]:
        """
        Método de ayuda para desglosar y formatear tags complejos.
        Maneja sufijos duales/múltiples separados por slashes, como 'A/B' o 'A/B/C'.
        El formato de entrada esperado es: 'BASE_CON_SUFIJO_A/SUFIJO_B/SUFIJO_C'.
        Ejemplo: 123-P-0101A/B/C
        """
        # --- LÓGICA DE DESGLOSE ACTUALIZADA PARA MÚLTIPLES SLASHES ---
        if '/' in tag:
            parts = tag.split('/')
            # La primera parte contiene la base y el primer sufijo, ej: '123-P-0101A'
            first_part = parts[0]
            
            # Extraer la base común y el primer sufijo
            base_match = re.match(r'^(.*[0-9])([A-Z])$', first_part)
            
            if base_match:
                tag_base = base_match.group(1) # ej: '123-P-0101'
                first_suffix = base_match.group(2) # ej: 'A'
                
                # Crear la lista de tags desglosados
                disaggregated_tags = [f"{tag_base}{first_suffix}"]
                
                # Añadir los tags para los sufijos restantes
                for suffix in parts[1:]:
                    if len(suffix) == 1 and suffix.isalpha():
                        disaggregated_tags.append(f"{tag_base}{suffix}")
                
                return disaggregated_tags

        # --- LÓGICA ANTERIOR PARA '\' (se mantiene por si acaso) ---
        # Si no se encontró '/', se puede verificar por '\' o cualquier otra lógica futura.
        # Por ahora, nos centramos en el requerimiento del '/'
        if '\\' in tag:
            # (Puedes mantener o adaptar la lógica anterior para '\' si aún es necesaria)
            pass

        # Si no es un tag compuesto, simplemente se devuelve en una lista de un elemento.
        return [tag]

    def extract_tags_from_text(self, text: str, page_number: int, document_path: Path) -> List[Dict[str, Any]]:
        """
        [VERSIÓN FINAL]
        Extrae y procesa tags de un texto. Incluye:
        1. Desglose de tags compuestos con múltiples slashes (A/B/C).
        2. Extracción del código de área.
        3. Adjuntar la taxonomía completa del CSV (Level 1, Level 2, Tag Code).
        4. Incluir el patrón regex que realizó la captura para debugging.
        """
        # --- LÓGICA EXISTENTE (desde la implementación de la taxonomía) ---
        all_matches_details = self.pattern_manager.find_all_matches(text)
        
        extracted_data = []
        
        for match_details in all_matches_details:
            original_tag_text = match_details['tag_capturado']
            # --- NUEVA LÓGICA: OBTENER EL REGEX USADO ---
            # Asumimos que el PatternManager ahora devuelve el regex en los detalles
            # (Se necesita una pequeña modificación en PatternManager que se muestra más abajo)
            regex_used = self.pattern_manager.taxonomy_data.get(match_details['pattern_name'], {}).get('regex', 'N/A')

            # Clasificación del tag según la taxonomía
            level1_class = match_details.get('clasificacion_level_1')

            # Desglosar tags si es necesario (ej: '...A/B' -> ['...A', '...B'])
            formatted_tags = self._disaggregate_and_format_tag(original_tag_text)

            # Iterar sobre cada tag desglosado para crear una fila de datos para cada uno
            for tag_to_format in formatted_tags:

                # --- INICIO DE LA NUEVA LÓGICA DE NORMALIZACIÓN ---
                
                # Por defecto, el tag formateado es el tag procesado.
                final_formatted_tag = tag_to_format
                
                # Regla de normalización: Si el tag es de clase 'Instrument' y contiene guiones,
                # se los quitamos para el campo 'tag_formateado'.
                # Ejemplo: '051-UXA-0012' -> '051UXA0012'
                if level1_class == 'Instrument' and '-' in tag_to_format:
                    final_formatted_tag = tag_to_format.replace('-', '')

                # --- FIN DE LA NUEVA LÓGICA DE NORMALIZACIÓN ---
                
                # Extraer el Área (los 3 primeros dígitos)
                area_code = None
                area_match = re.match(r'^(\d{3})', tag_to_format)
                if area_match:
                    area_code = area_match.group(1)

                # Extraer sufijo y base del tag ya formateado
                tag_base = tag_to_format
                sufijo = None
                # La lógica del sufijo busca la última letra que no esté precedida por otra letra
                match_sufijo = re.search(r'([^A-Z])([A-Z])$', final_formatted_tag)
                if match_sufijo:
                    sufijo = match_sufijo.group(2)
                    tag_base = final_formatted_tag[:-1]
                
                # --- LÓGICA EXISTENTE MEJORADA CON LOS NUEVOS CAMPOS ---
                tag_info = {
                    'documento_origen': str(document_path.name),
                    'area': area_code,
                    'tag_capturado': original_tag_text,
                    'tag_formateado': final_formatted_tag,
                    'tag_base': tag_base,
                    'sufijo_equipo': sufijo,
                    'clasificacion_level_1': match_details.get('clasificacion_level_1'),
                    'clasificacion_level_2': match_details.get('clasificacion_level_2'),
                    'tag_code': match_details.get('tag_code'),
                    'pagina_encontrada': page_number,
                    'contexto_linea': match_details.get('context'),
                    'regex_captura': regex_used,  # <-- CAMPO PARA DEBUGGING
                    'posicion_inicio': match_details.get('start'),
                    'posicion_fin': match_details.get('end'),
                }
                
                extracted_data.append(tag_info)

        return extracted_data

class CSVWriter:
    """Optimized CSV output handling."""
    
    def __init__(self, output_directory: Path = Path("output")):
        self.output_directory = output_directory
        self.output_directory.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def write_results(self, results: List[Dict[str, Any]], output_filename: str) -> Path:
        """
        [VERSIÓN FINAL]
        Escribe los resultados de la extracción en un archivo CSV, incluyendo
        todas las columnas de taxonomía, formato y debugging.
        """
        output_path = self.output_directory / output_filename
        
        # --- LÓGICA EXISTENTE MEJORADA ---
        # Definimos el orden final y completo de las columnas en el CSV de salida.
        fieldnames = [
            'documento_origen',
            'area',
            'tag_capturado',
            'tag_formateado',
            'tag_base',
            'sufijo_equipo',
            'clasificacion_level_1',
            'clasificacion_level_2',
            'tag_code',
            'pagina_encontrada',
            'contexto_linea',
            'regex_captura',  # <-- COLUMNA AÑADIDA
            'posicion_inicio',
            'posicion_fin'
        ]

        if not results:
            self.logger.warning("No hay datos para escribir en el CSV. Se creará un archivo vacío con cabeceras.")
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            return output_path
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                
                # La lógica de escritura por lotes se mantiene igual
                batch_size = 1000
                for i in range(0, len(results), batch_size):
                    batch = results[i:i + batch_size]
                    writer.writerows(batch)
                    csvfile.flush()
            
            self.logger.info(f"Se escribieron {len(results)} registros exitosamente en {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error escribiendo el archivo CSV: {e}")
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
        pattern_manager: Optional[RegexPatternManager] = None,
        db_filename: str = "extraction_results.db" 
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

        # Logica de DB
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        self.db_path = self.output_directory / db_filename
        self.db_table_name = "extracted_tags"
        
        # Definir las columnas una sola vez para mantener consistencia entre CSV y DB
        self.fieldnames = [
            'documento_origen', 'area', 'tag_capturado', 'tag_formateado', 'tag_base', 'sufijo_equipo',
            'clasificacion_level_1', 'clasificacion_level_2', 'tag_code', 'pagina_encontrada',
            'contexto_linea', 'regex_captura', 'posicion_inicio', 'posicion_fin']

# --- INICIO DE LAS NUEVAS FUNCIONES PARA SQLITE ---

    def initialize_database(self):
        """
        Crea el archivo de base de datos SQLite y la tabla 'extracted_tags' si no existen.
        Define la llave primaria compuesta. Se debe llamar una vez por ejecución.
        """
        try:
            self.logger.info(f"Inicializando la base de datos en: {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Crear la tabla con los tipos de datos adecuados y la llave primaria compuesta
            # Usamos "TEXT" para la mayoría de los campos para máxima flexibilidad.
            # La llave primaria en (tag_formateado, documento_origen) previene duplicados.
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.db_table_name} (
                documento_origen TEXT,
                area TEXT,
                tag_capturado TEXT,
                tag_formateado TEXT NOT NULL,
                tag_base TEXT,
                sufijo_equipo TEXT,
                clasificacion_level_1 TEXT,
                clasificacion_level_2 TEXT,
                tag_code TEXT,
                pagina_encontrada INTEGER,
                contexto_linea TEXT,
                regex_captura TEXT,
                posicion_inicio INTEGER,
                posicion_fin INTEGER,
                PRIMARY KEY ("tag_formateado", "documento_origen")
            );
            """
            cursor.execute(create_table_sql)
            conn.commit()
            conn.close()
            self.logger.info(f"La tabla '{self.db_table_name}' está lista.")
        except Exception as e:
            self.logger.error(f"Error al inicializar la base de datos: {e}")
            raise

    def save_results_to_sqlite(self, results: List[Dict[str, Any]]):
        """
        Guarda una lista de resultados de tags en la base de datos SQLite.
        Utiliza 'INSERT OR REPLACE' para manejar duplicados, actualizando el registro si ya existe.
        """
        if not results:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Crear el statement de inserción parametrizado para evitar inyección SQL
            # Usamos INSERT OR REPLACE para que si un tag de un documento ya existe, se actualice.
            # Esto es útil si se re-procesa un archivo.
            columns = ', '.join(self.fieldnames)
            placeholders = ', '.join(['?'] * len(self.fieldnames))
            sql = f"INSERT OR REPLACE INTO {self.db_table_name} ({columns}) VALUES ({placeholders})"
            
            # Preparar los datos como una lista de tuplas en el orden correcto
            data_to_insert = []
            for row_dict in results:
                # Se asegura de que cada valor esté en el orden correcto de self.fieldnames
                row_tuple = tuple(row_dict.get(field) for field in self.fieldnames)
                data_to_insert.append(row_tuple)

            cursor.executemany(sql, data_to_insert)
            conn.commit()
            conn.close()
            self.logger.info(f"Guardados {len(results)} registros en la base de datos SQLite.")

        except Exception as e:
            self.logger.error(f"Error al guardar resultados en SQLite: {e}")
            # No relanzamos el error para no detener el procesamiento de los demás archivos        
    
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
            
            # if file_size_mb > 500:  # 500MB limit
                # raise PDFExtractionError(
                    # f"File size {file_size_mb:.2f}MB exceeds maximum limit of 500MB",
                    # error_code="FILE_TOO_LARGE"
                # )
            
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