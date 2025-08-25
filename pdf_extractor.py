#!/usr/bin/env python3
"""
Enterprise-grade PDF Data Extractor for Technical Equipment Tag Recognition
Author: Enterprise Development Team
Version: 2.4.0 (Enhanced Equipment Tag Disaggregation)
Python: 3.8+
"""

import re
import csv
import gc
import os
import time
import psutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Union
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymupdf

# Importar PatternManager desde su ubicación correcta
from utils.pattern_manager import PatternManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)

# --- Clases de Excepciones y Datos (Sin Cambios) ---

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
        super().__init__(f"PDF file not found: {file_path}", "PDF_FILE_NOT_FOUND", {"file_path": str(file_path)})

class PDFCorruptedError(PDFExtractionError):
    """Raised when PDF file is corrupted or unreadable."""
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

# --- Clases de Utilidad (Sin Cambios) ---

class MemoryMonitor:
    """Memory usage monitoring and management."""
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.peak_memory = 0
    
    def get_current_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024
    
    def check_memory_threshold(self) -> bool:
        current_memory = self.get_current_memory_mb()
        self.peak_memory = max(self.peak_memory, current_memory)
        if current_memory > self.max_memory_mb:
            gc.collect()
            if self.get_current_memory_mb() > self.max_memory_mb:
                raise MemoryError(f"Memory usage {current_memory:.1f}MB exceeds limit {self.max_memory_mb}MB")
        return current_memory > self.max_memory_mb * 0.8

class PDFReader:
    """Optimized PDF reading implementation using PyMuPDF."""
    def __init__(self, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def open_pdf(self, pdf_path: Path):
        doc = None
        try:
            doc = pymupdf.open(str(pdf_path))
            if doc.needs_pass:
                raise PDFExtractionError(f"PDF is password protected: {pdf_path}", "PASSWORD_PROTECTED")
            yield doc
        except (pymupdf.FileNotFoundError, RuntimeError):
            raise PDFFileNotFoundError(pdf_path)
        except Exception:
            raise PDFCorruptedError(f"PDF file appears to be corrupted: {pdf_path}")
        finally:
            if doc:
                doc.close()
            gc.collect()
    
    def extract_text_chunked(self, pdf_path: Path) -> Iterator[tuple[str, int]]:
        with self.open_pdf(pdf_path) as doc:
            for page_num in range(doc.page_count):
                try:
                    yield doc[page_num].get_text(), page_num + 1
                except Exception as e:
                    self.logger.warning(f"Error reading page {page_num + 1}: {e}")

# --- Motor de Lógica de Extracción (Lógica Principal Actualizada) ---

class DataExtractor:
    """Core data extraction logic with advanced post-processing."""
    def __init__(self, pattern_manager: PatternManager):
        self.pattern_manager = pattern_manager
        self.logger = logging.getLogger(__name__)

    def _normalize_tag(self, tag: str, level1: str) -> str:
        if level1 == 'Instrument' and '-' in tag:
            return tag.replace('-', '')
        if ' ' in tag:
            return re.sub(r'(\d{3})\s+([A-Z]{1,4})', r'\1\2', tag)
        return tag

    def _disaggregate_tag(self, tag: str) -> List[str]:
        """
        Desagrega tags con formato A/B/C/D en tags individuales
        Maneja múltiples formatos:
        - TAG123A/B/C → TAG123A, TAG123B, TAG123C
        - TAG-123-A/B/C → TAG-123-A, TAG-123-B, TAG-123-C
        - TAG123/A/B/C → TAG123A, TAG123B, TAG123C
        - 062E6211A/B/C → 062E6211A, 062E6211B, 062E6211C
        """
        if '/' not in tag:
            return [tag]
        
        # Método 1: Buscar si hay un patrón letra/letra al final (más común)
        # Ejemplo: 062E6211A/B/C o TAG-123-A/B/C
        pattern1 = r'^(.*?)([A-Z])(/[A-Z])+$'
        match1 = re.match(pattern1, tag)
        
        if match1:
            # Extraer la base y la primera letra
            base = match1.group(1)  # "062E6211" o "TAG-123-"
            first_letter = match1.group(2)  # "A"
            
            # Obtener todas las letras
            rest_part = tag[len(base) + len(first_letter):]  # "/B/C"
            other_letters = [l for l in rest_part.split('/') if l]  # ["B", "C"]
            
            # Combinar todas las letras
            all_letters = [first_letter] + other_letters
            
            # Crear los tags desagregados
            return [base + letter for letter in all_letters]
        
        # Método 2: Buscar si hay un patrón /letra/letra (menos común)
        # Ejemplo: TAG123/A/B/C
        pattern2 = r'^(.*?)/([A-Z](/[A-Z])+)$'
        match2 = re.match(pattern2, tag)
        
        if match2:
            base = match2.group(1)  # "TAG123"
            letters_part = match2.group(2)  # "A/B/C"
            letters = [l for l in letters_part.split('/') if l]  # ["A", "B", "C"]
            
            # Si la base termina con guión, mantenerlo
            if base.endswith('-'):
                return [base + letter for letter in letters]
            else:
                # Si no hay guión, agregar las letras directamente
                return [base + letter for letter in letters]
        
        # Método 3: Manejo especial para patrones con múltiples partes
        # Ejemplo: 062-E-6211-A/B/C donde necesitamos mantener la estructura
        parts = tag.split('/')
        if len(parts) > 1:
            # Verificar si la primera parte termina con una letra mayúscula
            first_part = parts[0]
            base_match = re.match(r'^(.*?)([A-Z])$', first_part)
            
            if base_match:
                # La primera parte termina con letra
                tag_base = base_match.group(1)  # "062-E-6211-" o "062E6211"
                first_suffix = base_match.group(2)  # "A"
                
                # Verificar que las otras partes son letras simples
                if all(len(p) == 1 and p.isalpha() and p.isupper() for p in parts[1:]):
                    # Todas las partes posteriores son letras simples
                    disaggregated = [tag_base + first_suffix]
                    for suffix in parts[1:]:
                        disaggregated.append(tag_base + suffix)
                    return disaggregated
            else:
                # La primera parte no termina con letra, verificar si las partes son letras simples
                if all(len(p) == 1 and p.isalpha() and p.isupper() for p in parts[1:]):
                    # Es un patrón como TAG123/A/B/C
                    base = first_part
                    # Si la base termina con guión, mantenerlo
                    if not base.endswith('-'):
                        # Agregar directamente
                        return [base + letter for letter in parts[1:]]
                    else:
                        # Con guión al final
                        return [base + letter for letter in parts[1:]]
        
        # Si no coincide con ningún patrón conocido, devolver el tag original
        self.logger.debug(f"Tag '{tag}' no coincide con patrones de desagregación conocidos")
        return [tag]

    def _create_base_tag_info(self, match_details: Dict, page_num: int, doc_path: Path) -> Dict:
        return {
            'documento_origen': doc_path.name,
            'pagina_encontrada': page_num,
            'tag_capturado': match_details['tag_capturado'],
            'clasificacion_level_1': match_details.get('clasificacion_level_1'),
            'clasificacion_level_2': match_details.get('clasificacion_level_2'),
            'tag_code': match_details.get('tag_code'),
            'regex_captura': self.pattern_manager.taxonomy_data.get(match_details['pattern_name'], {}).get('Regex Pattern', 'N/A'),
            'contexto_linea': match_details.get('context'),
            'posicion_inicio': match_details.get('start'),
            'posicion_fin': match_details.get('end'),
        }

    def extract_tags_from_text(self, text: str, page_number: int, document_path: Path) -> List[Dict[str, Any]]:
        all_matches = self.pattern_manager.find_all_matches(text)
        extracted_data = []
        doc_name_without_ext = document_path.stem

        for match in all_matches:
            if match['clasificacion_level_1'] == 'Document' and doc_name_without_ext in match['tag_capturado']:
                continue

            # Aplicar desagregación para Equipos e Instrumentos
            # MEJORA: Ahora la desagregación se aplica específicamente a Equipment e Instrument
            if match['clasificacion_level_1'] in ['Equipment', 'Instrument']:
                disaggregated_tags = self._disaggregate_tag(match['tag_capturado'])
                self.logger.debug(f"Desagregando {match['clasificacion_level_1']} tag '{match['tag_capturado']}' en: {disaggregated_tags}")
            else:
                disaggregated_tags = [match['tag_capturado']]
            
            for tag_instance in disaggregated_tags:
                info = self._create_base_tag_info(match, page_number, document_path)
                info['tag_formateado'] = self._normalize_tag(tag_instance, info['clasificacion_level_1'])
                
                # --- INICIO DE LA CORRECCIÓN: CONSOLIDACIÓN DE DATOS ---
                
                # Regla general para 'area': se extrae de los 3 primeros dígitos.
                info['area'] = re.match(r'^(\d{3})', tag_instance.replace(" ", "")).group(1) if re.match(r'^(\d{3})', tag_instance.replace(" ", "")) else None

                match_obj = match.get('match_object')
                if not match_obj:
                    extracted_data.append(info)
                    continue

                # Lógica de desglose específica por tipo de tag
                level1_class = info['clasificacion_level_1']
                
                if level1_class == 'ElectricalEquipment':
                    groups = match_obj.groups()
                    info['tag_code'] = groups[0] if groups else ''
                    info['tag_base'] = groups[1] if len(groups) > 1 else '' # Sequence Number es la base
                    info['sufijo_equipo'] = "".join(groups[2:]) if len(groups) > 2 else ''
                
                elif level1_class == 'Piping':
                    try:
                        # La 'area' ya fue capturada arriba. No necesitamos 'piping_unit'.
                        info['piping_npd'] = match_obj.group('NPD')
                        info['piping_fluid_code'] = match_obj.group('Fluid_Code')
                        info['piping_sequence'] = match_obj.group('Sequence')
                        info['piping_class'] = match_obj.group('Piping_Class')
                        info['piping_insulation'] = match_obj.group('Insulation') or ''
                    except IndexError:
                        self.logger.warning(f"Patrón de Piping sin grupos nombrados para '{match['tag_capturado']}'.")
                else:
                    # Lógica general para 'tag_base' y 'sufijo_equipo' para otros tipos (incluido Equipment)
                    tag_base = info['tag_formateado']
                    sufijo = None
                    sufijo_match = re.search(r'([A-Z])$', tag_base)
                    if sufijo_match and (len(tag_base) < 5 or not tag_base[-2].isalpha()):
                        sufijo = sufijo_match.group(1)
                        tag_base = tag_base[:-1]
                    info['tag_base'] = tag_base
                    info['sufijo_equipo'] = sufijo

                # --- FIN DE LA CORRECCIÓN ---
                
                extracted_data.append(info)
        return extracted_data

# --- Gestor de Salida a CSV ---
# --- Gestor de Salida a CSV (Simplificado) ---

class CSVWriter:
    def __init__(self, output_directory: Path):
        self.output_directory = output_directory
        self.logger = logging.getLogger(__name__)

    def write_results(self, results: List[Dict[str, Any]], output_filename: str, fieldnames: List[str]) -> Path:
        output_path = self.output_directory / output_filename
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            self.logger.info(f"Se escribieron {len(results)} registros en {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error escribiendo el archivo CSV: {e}")
            raise

# --- Clase Principal del Extractor (Integrador de Componentes) ---

class PDFDataExtractor:
    def __init__(
        self,
        chunk_size: int = 10,
        max_memory_mb: int = 1024,
        extraction_timeout: int = 300,
        output_directory: str = "output",
        pattern_manager: Optional[PatternManager] = None
    ):
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.extraction_timeout = extraction_timeout
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        self.pattern_manager = pattern_manager if pattern_manager is not None else PatternManager()
        self.pdf_reader = PDFReader(chunk_size=chunk_size)
        self.data_extractor = DataExtractor(self.pattern_manager)
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        self.logger = logging.getLogger(__name__)

        # Centralización de la definición de columnas de salida
        self.fieldnames = [
            'documento_origen',
            'tag_capturado',
            'tag_formateado',
            'tag_base',
            'area',
            'sufijo_equipo',
            'clasificacion_level_1',
            'clasificacion_level_2',
            'tag_code',
            'piping_npd',
            'piping_fluid_code',
            'piping_sequence',
            'piping_class',
            'piping_insulation',
            'pagina_encontrada',
            'contexto_linea',
            'posicion_inicio',
            'posicion_fin',
            'regex_captura'
        ]
        
        self.csv_writer = CSVWriter(self.output_directory)
        
        # Configuración de la base de datos
        db_filename = "extraction_results.db"
        self.db_path = self.output_directory / db_filename
        self.db_table_name = "extracted_tags"
        
    def initialize_database(self):
        """Crea la tabla de la base de datos usando la lista de fieldnames centralizada."""
        try:
            self.logger.info(f"Inicializando la base de datos en: {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            columns_definitions = []
            for field in self.fieldnames:
                col_type = "INTEGER" if "pagina" in field or "posicion" in field else "TEXT"
                columns_definitions.append(f'"{field}" {col_type}')
            
            pk = 'PRIMARY KEY ("tag_formateado", "documento_origen")'
            
            create_table_sql = f'CREATE TABLE IF NOT EXISTS "{self.db_table_name}" ({", ".join(columns_definitions)}, {pk});'
            
            cursor.execute(create_table_sql)
            conn.commit()
            conn.close()
            self.logger.info(f"La tabla '{self.db_table_name}' está lista.")
        except Exception as e:
            self.logger.error(f"Error al inicializar la base de datos: {e}")
            raise

    def save_results_to_sqlite(self, results: List[Dict[str, Any]]):
        """Guarda los resultados en SQLite usando la lista de fieldnames centralizada."""
        if not results: return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            columns = ', '.join(f'"{f}"' for f in self.fieldnames)
            placeholders = ', '.join(['?'] * len(self.fieldnames))
            sql = f'INSERT OR REPLACE INTO "{self.db_table_name}" ({columns}) VALUES ({placeholders})'
            
            data_to_insert = [tuple(row.get(field) for field in self.fieldnames) for row in results]
            
            cursor.executemany(sql, data_to_insert)
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error al guardar resultados en SQLite: {e}")

    def extract_from_single_pdf(self, pdf_path: Union[str, Path]) -> ExtractionResult:
        pdf_path = Path(pdf_path)
        start_time = time.time()
        
        try:
            if not pdf_path.exists():
                raise PDFFileNotFoundError(pdf_path)

            self.logger.info(f"Starting extraction for: {pdf_path} ({pdf_path.stat().st_size / (1024*1024):.2f} MB)")
            all_extracted_data = []
            page_count = 0
            
            for text_chunk, page_number in self.pdf_reader.extract_text_chunked(pdf_path):
                page_count = page_number
                if self.memory_monitor.check_memory_threshold():
                    self.logger.warning(f"Memory usage approaching limit on page {page_number}")
                
                chunk_data = self.data_extractor.extract_tags_from_text(text_chunk, page_number, pdf_path)
                all_extracted_data.extend(chunk_data)
            
            processing_time = time.time() - start_time
            metrics = {
                'page_count': page_count, 'peak_memory_mb': self.memory_monitor.peak_memory
            }
            self.logger.info(f"Extraction completed: {len(all_extracted_data)} tags in {processing_time:.2f}s")
            return ExtractionResult(pdf_path, all_extracted_data, metrics, True, processing_time=processing_time)
            
        except Exception as e:
            self.logger.error(f"Extraction failed for {pdf_path}: {e}")
            return ExtractionResult(pdf_path, [], {}, False, str(e), time.time() - start_time)
    
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
        
        return self.csv_writer.write_results(all_data, output_filename, self.fieldnames)
    
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