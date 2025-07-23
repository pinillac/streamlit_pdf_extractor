"""
PDF Processor Component
Core processing engine that integrates with the PDFDataExtractor
"""

import streamlit as st
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Any, Optional, Union
import time
import logging
#from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Import the original PDF extractor
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pdf_extractor import PDFDataExtractor
from utils.pattern_manager import PatternManager

class PDFProcessor:
    """Enhanced PDF processor with Streamlit integration"""
    
    def __init__(self):
        self.extractor = None
        self.pattern_manager = PatternManager() # <--- USA SIEMPRE EL GESTOR MODERNO
        self.custom_patterns = {}
        self.logger = logging.getLogger(__name__)
        self.processing_stats = {
            'files_processed': 0,
            'total_tags': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the processor with given settings
        
        Args:
            config: Configuration dictionary
        """
        # Initialize extractor with configuration
        self.extractor = PDFDataExtractor(
            chunk_size=config.get('chunk_size', 10),
            max_memory_mb=config.get('memory_limit_mb', 1024),
            extraction_timeout=config.get('processing_timeout', 300),
            output_directory=tempfile.gettempdir(),
            pattern_manager=self.pattern_manager # <--- AQUÍ ESTÁ LA MAGIA
        )
        
        # Load custom patterns if provided
        if config.get('use_custom_patterns') and config.get('custom_patterns'):
            self.load_custom_patterns(config['custom_patterns'])
        
        # Set max workers for parallel processing
        self.max_workers = config.get('max_workers', 4)
    
    def load_custom_patterns(self, patterns_json: str):
        """
        Load and validate custom regex patterns
        """
        try:
            custom_patterns = json.loads(patterns_json)
            
            # Como ahora el pattern_manager es parte del PDFProcessor, podemos usarlo directamente.
            # Ya no necesitamos comprobar 'hasattr(self.extractor, 'pattern_manager')'
            
            for pattern_name, pattern_regex in custom_patterns.items():
                # --- INICIO DE LA CORRECCIÓN ---
                
                # Usamos el método oficial del gestor de patrones para añadir el nuevo patrón.
                # Este método se encarga de compilarlo y guardarlo correctamente.
                # Pasamos 'custom' como categoría para los patrones del usuario.
                success = self.pattern_manager.add_pattern(
                    name=pattern_name,
                    regex=pattern_regex,
                    category='custom'
                )
                
                if success:
                    self.custom_patterns[pattern_name] = pattern_regex
                else:
                    # El método add_pattern ya registra el error, pero podemos añadir un log aquí si queremos.
                    self.logger.warning(f"Could not add custom pattern: {pattern_name}")
                    
                # --- FIN DE LA CORRECCIÓN ---
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON for custom patterns: {e}")
    
    def process_file(self, file) -> Dict[str, Any]:
        """
        Process a single file
        
        Args:
            file: Streamlit uploaded file or file path
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'success': False,
            'data': [],
            'error': None,
            'metrics': {}
        }
        
        try:
            # --- INICIO DE LA MODIFICACIÓN ---
            if hasattr(file, 'path'):  # Es nuestro objeto LocalFile
                file_path = Path(file.path)
            elif hasattr(file, 'read'):  # Es un archivo subido por Streamlit
                temp_path = Path(tempfile.gettempdir()) / file.name
                with open(temp_path, 'wb') as f:
                    f.write(file.read())
                file_path = temp_path
                file.seek(0)
            else:
                raise TypeError("Unsupported file object type.")
            # --- FIN DE LA MODIFICACIÓN ---
        
        # try:
            # # Save uploaded file to temp location if needed
            # if hasattr(file, 'read'):  # Streamlit uploaded file
                # temp_path = Path(tempfile.gettempdir()) / file.name
                # with open(temp_path, 'wb') as f:
                    # f.write(file.read())
                # file_path = temp_path
                # file.seek(0)  # Reset file pointer
            # else:
                # file_path = Path(file)
            
            # Process with extractor
            extraction_result = self.extractor.extract_from_single_pdf(file_path)
            
            if extraction_result.success:
                result['success'] = True
                result['data'] = extraction_result.extracted_data
                result['metrics'] = extraction_result.processing_metrics
                
                # Update processing stats
                self.processing_stats['files_processed'] += 1
                self.processing_stats['total_tags'] += len(extraction_result.extracted_data)
            else:
                result['error'] = extraction_result.error_message
                self.processing_stats['errors'] += 1
            
            # Cleanup temp file if created
            if hasattr(file, 'read') and temp_path.exists():
                temp_path.unlink()
                
        except Exception as e:
            result['error'] = str(e)
            self.processing_stats['errors'] += 1
            self.logger.error(f"Error processing file: {e}")
        
        finally:
            # Force garbage collection after processing
            gc.collect()
        
        return result
        
    def process_ab_pairs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
         """
         Post-procesa la lista de resultados para encontrar y vincular
         equipos duales (pares A/B).
 
         Args:
             results: Lista de diccionarios de tags extraídos.
 
         Returns:
             La lista de resultados enriquecida con información de pares.
         """
         # Mapa para agrupar tags por su base (e.g., '011P0001' -> [tagA, tagB])
         base_tag_map = defaultdict(list)
         
         # 1. Agrupar tags por su 'tag_base'
         for i, res in enumerate(results):
             if res.get('tag_base'):
                 base_tag_map[res['tag_base']].append(i) # Guardar el índice del resultado
 
         # 2. Iterar sobre los grupos para encontrar pares
         for base_tag, indices in base_tag_map.items():
             if len(indices) == 2: # Se encontró un par A/B
                 idx_a, idx_b = indices[0], indices[1]
 
                 # Asegurarse de que uno es 'A' y el otro 'B'
                 tag_a = results[idx_a]
                 tag_b = results[idx_b]
 
                 if (tag_a.get('sufijo_equipo') == 'A' and tag_b.get('sufijo_equipo') == 'B') or \
                    (tag_a.get('sufijo_equipo') == 'B' and tag_b.get('sufijo_equipo') == 'A'):
                     
                     # Vincular los tags entre sí
                     tag_a['sibling_tag'] = tag_b['tag_capturado']
                     tag_b['sibling_tag'] = tag_a['tag_capturado']
 
         return results
 
    def get_processing_stats(self) -> Dict[str, Any]:
         """Get current processing statistics"""
         stats = self.processing_stats.copy()    
    
    def process_multiple_files(self, files: List[Any], 
                             progress_callback=None) -> List[Dict[str, Any]]:
        """
        Process multiple files with parallel processing
        
        Args:
            files: List of files to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processing results
        """
        results = []
        self.processing_stats['start_time'] = time.time()
        
        # Determine optimal number of workers
        num_workers = min(self.max_workers, len(files))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self.process_file, file): file 
                for file in files
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    result['filename'] = file.name if hasattr(file, 'name') else str(file)
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        'success': False,
                        'filename': file.name if hasattr(file, 'name') else str(file),
                        'data': [],
                        'error': str(e)
                    })
                
                completed += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed, len(files))
        
        self.processing_stats['end_time'] = time.time()
        
        return results
    
    def process_folder(self, folder_path: Path, 
                      pattern: str = "*.pdf",
                      recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a folder
        
        Args:
            folder_path: Path to folder
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of processing results
        """
        # Find all PDF files
        if recursive:
            pdf_files = list(folder_path.rglob(pattern))
        else:
            pdf_files = list(folder_path.glob(pattern))
        
        if not pdf_files:
            return []
        
        # Process files
        return self.process_multiple_files(pdf_files)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['start_time'] and stats['end_time']:
            stats['total_time'] = stats['end_time'] - stats['start_time']
            stats['files_per_second'] = stats['files_processed'] / stats['total_time'] if stats['total_time'] > 0 else 0
            stats['tags_per_second'] = stats['total_tags'] / stats['total_time'] if stats['total_time'] > 0 else 0
        
        return stats
    
    def validate_extraction_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate and analyze extraction results
        
        Args:
            results: List of extraction results
            
        Returns:
            Validation report
        """
        validation_report = {
            'total_results': len(results),
            'unique_tags': set(),
            'tag_distribution': {},
            'duplicate_tags': {},
            'suspicious_patterns': [],
            'quality_score': 0.0
        }
        
        all_tags = []
        
        # Collect all tags
        for result in results:
            if isinstance(result, dict) and 'tag_capturado' in result:
                tag = result['tag_capturado']
                equipment_type = result.get('tipo_elemento', 'unknown')
                
                all_tags.append(tag)
                validation_report['unique_tags'].add(tag)
                
                # Count distribution
                if equipment_type not in validation_report['tag_distribution']:
                    validation_report['tag_distribution'][equipment_type] = 0
                validation_report['tag_distribution'][equipment_type] += 1
        
        # Find duplicates
        from collections import Counter
        tag_counts = Counter(all_tags)
        validation_report['duplicate_tags'] = {
            tag: count for tag, count in tag_counts.items() if count > 1
        }
        
        # Check for suspicious patterns
        for tag in validation_report['unique_tags']:
            # Check for common issues
            if len(tag) < 3:
                validation_report['suspicious_patterns'].append({
                    'tag': tag,
                    'issue': 'Tag too short'
                })
            elif len(tag) > 20:
                validation_report['suspicious_patterns'].append({
                    'tag': tag,
                    'issue': 'Tag unusually long'
                })
        
        # Calculate quality score
        total_tags = len(all_tags)
        unique_tags = len(validation_report['unique_tags'])
        
        if total_tags > 0:
            uniqueness_ratio = unique_tags / total_tags
            distribution_score = len(validation_report['tag_distribution']) / 7  # Expected 7 types
            suspicious_ratio = 1 - (len(validation_report['suspicious_patterns']) / total_tags)
            
            validation_report['quality_score'] = (
                uniqueness_ratio * 0.4 + 
                distribution_score * 0.3 + 
                suspicious_ratio * 0.3
            ) * 100
        
        return validation_report
    
    def merge_results(self, results_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Merge results from multiple processing runs
        
        Args:
            results_list: List of result lists
            
        Returns:
            Merged results with deduplication
        """
        merged = []
        seen_tags = {}  # Track tag occurrences
        
        for results in results_list:
            for result in results:
                tag_key = f"{result.get('tag_capturado', '')}_{result.get('documento_origen', '')}"
                
                if tag_key not in seen_tags:
                    # First occurrence
                    result['occurrence_count'] = 1
                    merged.append(result)
                    seen_tags[tag_key] = len(merged) - 1
                else:
                    # Duplicate - increment count
                    merged[seen_tags[tag_key]]['occurrence_count'] += 1
        
        return merged
    
    def apply_post_processing_filters(self, results: List[Dict[str, Any]], 
                                    filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply post-processing filters to results
        
        Args:
            results: Raw extraction results
            filters: Filter criteria
            
        Returns:
            Filtered results
        """
        filtered = results.copy()
        
        # Filter by equipment type
        if filters.get('equipment_types'):
            filtered = [
                r for r in filtered 
                if r.get('tipo_elemento') in filters['equipment_types']
            ]
        
        # Filter by tag pattern
        if filters.get('tag_pattern'):
            import re
            pattern = re.compile(filters['tag_pattern'])
            filtered = [
                r for r in filtered 
                if pattern.match(r.get('tag_capturado', ''))
            ]
        
        # Filter by page range
        if filters.get('page_range'):
            min_page, max_page = filters['page_range']
            filtered = [
                r for r in filtered 
                if min_page <= r.get('pagina_encontrada', 0) <= max_page
            ]
        
        # Filter by document
        if filters.get('documents'):
            filtered = [
                r for r in filtered 
                if r.get('documento_origen') in filters['documents']
            ]
        
        return filtered
    
    def generate_processing_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive processing report
        
        Args:
            results: Processing results
            
        Returns:
            Processing report
        """
        report = {
            'summary': {
                'total_files': self.processing_stats['files_processed'],
                'total_tags': self.processing_stats['total_tags'],
                'total_errors': self.processing_stats['errors'],
                'processing_time': 0
            },
            'performance': {},
            'tag_statistics': {},
            'file_statistics': {},
            'recommendations': []
        }
        
        # Calculate processing time
        if self.processing_stats['start_time'] and self.processing_stats['end_time']:
            report['summary']['processing_time'] = (
                self.processing_stats['end_time'] - self.processing_stats['start_time']
            )
        
        # Performance metrics
        if report['summary']['processing_time'] > 0:
            report['performance'] = {
                'files_per_second': (
                    report['summary']['total_files'] / report['summary']['processing_time']
                ),
                'tags_per_second': (
                    report['summary']['total_tags'] / report['summary']['processing_time']
                ),
                'average_tags_per_file': (
                    report['summary']['total_tags'] / report['summary']['total_files']
                    if report['summary']['total_files'] > 0 else 0
                )
            }
        
        # Tag statistics
        tag_types = {}
        for result in results:
            if result.get('success') and result.get('data'):
                for tag in result['data']:
                    tag_type = tag.get('tipo_elemento', 'unknown')
                    tag_types[tag_type] = tag_types.get(tag_type, 0) + 1
        
        report['tag_statistics'] = tag_types
        
        # File statistics
        file_stats = []
        for result in results:
            if result.get('metrics'):
                file_stats.append({
                    'filename': result.get('filename', 'unknown'),
                    'pages': result['metrics'].get('page_count', 0),
                    'tags': len(result.get('data', [])),
                    'processing_time': result['metrics'].get('processing_time_seconds', 0),
                    'memory_used': result['metrics'].get('peak_memory_mb', 0)
                })
        
        report['file_statistics'] = file_stats
        
        # Generate recommendations
        if report['summary']['total_errors'] > 0:
            report['recommendations'].append(
                f"⚠️ {report['summary']['total_errors']} files failed processing. Check error logs."
            )
        
        if report['performance'].get('files_per_second', 0) < 0.1:
            report['recommendations'].append(
                "🐌 Processing speed is slow. Consider increasing chunk size or memory limit."
            )
        
        if report['performance'].get('average_tags_per_file', 0) < 10:
            report['recommendations'].append(
                "📉 Low tag extraction rate. Verify PDF content and pattern configuration."
            )
        
        return report