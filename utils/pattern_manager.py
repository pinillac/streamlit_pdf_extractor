"""
Pattern Manager Utility
Manages regex patterns for equipment tag extraction
"""

import re
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Pattern
import streamlit as st
from pathlib import Path
import logging
from collections import defaultdict

# Try to import Levenshtein, use a fallback if not available
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    # Simple fallback for similarity calculation
    def simple_similarity(s1, s2):
        """Simple similarity calculation without Levenshtein"""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        # Calculate common characters ratio
        common = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
        max_len = max(len(s1), len(s2))
        return common / max_len if max_len > 0 else 0.0

class PatternManager:
    def __init__(self, taxonomy_file='taxonomy_1.csv'):
        self.logger = logging.getLogger(__name__)
        # self.patterns almacena ahora el regex compilado
        self.compiled_patterns = {} 
        # self.taxonomy_data almacena toda la info de la taxonomía por nombre de patrón
        self.taxonomy_data = {}
        self.load_patterns_from_csv(taxonomy_file)

    def load_patterns_from_csv(self, taxonomy_file: str):
        """
        Carga los patrones y su taxonomía desde un archivo CSV.
        Esto reemplaza la lógica de patrones 'hard-coded'.
        """
        try:
            # Asumimos que taxonomy_1.csv está en la misma carpeta o en una ruta accesible
            taxonomy_path = Path(__file__).parent.parent / taxonomy_file
            df = pd.read_csv(taxonomy_path)
            
            # Limpiar nombres de columna (quitar espacios)
            df.columns = df.columns.str.strip()

            # Generar un nombre único para cada patrón/regla
            df['pattern_name'] = df['Level 1'] + '_' + df['Level 2']

            for index, row in df.iterrows():
                pattern_name = row['pattern_name']
                regex_str = row['Regex Pattern']

                if pd.isna(regex_str) or not isinstance(regex_str, str) or regex_str.strip() == '':
                    # Si no hay regex, se podría intentar generar uno, o simplemente ignorarlo por ahora
                    continue
                
                try:
                    compiled = re.compile(regex_str)
                    self.compiled_patterns[pattern_name] = compiled
                    
                    # Guardar toda la metadata de la taxonomía
                    self.taxonomy_data[pattern_name] = {
                        'level_1': row['Level 1'],
                        'level_2': row['Level 2'],
                        'tag_code': row['Tag Code'],
                        'regex': regex_str
                    }
                except re.error as e:
                    self.logger.error(f"Error compilando regex para '{pattern_name}': {regex_str} -> {e}")

            self.logger.info(f"Cargados {len(self.compiled_patterns)} patrones desde {taxonomy_file}")

        except FileNotFoundError:
            self.logger.error(f"Archivo de taxonomía no encontrado: {taxonomy_path}. Cargando patrones por defecto.")
            # Aquí podrías poner tu antiguo `load_default_patterns` como fallback
            self.load_default_patterns() # Asegúrate que esta función exista si quieres el fallback
        except Exception as e:
            self.logger.error(f"Error procesando el archivo de taxonomía: {e}")
    
    def add_pattern(self, name: str, regex: str, category: str = 'custom') -> bool:
        """
        Add a new pattern
        
        Args:
            name: Pattern name
            regex: Regular expression
            category: Pattern category
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate and compile pattern
            compiled = re.compile(regex)
            
            self.patterns[name] = {
                'regex': regex,
                'category': category,
                'compiled': compiled
            }
            self.compiled_patterns[name] = compiled
            
            self.logger.info(f"Added pattern '{name}' in category '{category}'")
            return True
            
        except re.error as e:
            self.logger.error(f"Invalid regex pattern '{name}': {e}")
            return False
    
    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern"""
        if name in self.patterns:
            del self.patterns[name]
            del self.compiled_patterns[name]
            self.logger.info(f"Removed pattern '{name}'")
            return True
        return False
    
    def validate_patterns(self, regex: str, test_strings: List[str] = None) -> Dict[str, Any]:
        """
        Validate a regex pattern
        
        Args:
            regex: Regular expression to validate
            test_strings: Optional test strings
            
        Returns:
            Validation results
        """
        result = {
            'valid': False,
            'error': None,
            'matches': [],
            'performance': {}
        }
        
        try:
            # Compile pattern
            import time
            start_time = time.time()
            pattern = re.compile(regex)
            compile_time = time.time() - start_time
            
            result['valid'] = True
            result['performance']['compile_time'] = compile_time
            
            # Test against strings if provided
            if test_strings:
                match_time_total = 0
                
                for test_str in test_strings:
                    start_time = time.time()
                    matches = list(pattern.finditer(test_str))
                    match_time = time.time() - start_time
                    match_time_total += match_time
                    
                    for match in matches:
                        result['matches'].append({
                            'text': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'groups': match.groups(),
                            'source': test_str[:50] + '...' if len(test_str) > 50 else test_str
                        })
                
                result['performance']['avg_match_time'] = match_time_total / len(test_strings) if test_strings else 0
            
            # Check for potential issues
            warnings = self.check_pattern_quality(regex)
            if warnings:
                result['warnings'] = warnings
                
        except re.error as e:
            result['error'] = str(e)
            
        return result
    
    def check_pattern_quality(self, regex: str) -> List[str]:
        """
        Check pattern for potential issues
        
        Args:
            regex: Regular expression
            
        Returns:
            List of warnings
        """
        warnings = []
        
        # Check for catastrophic backtracking
        if re.search(r'(\.\*){2,}', regex):
            warnings.append("Pattern contains multiple .* which may cause performance issues")
        
        if re.search(r'(\+\?|\*\?){2,}', regex):
            warnings.append("Pattern contains nested quantifiers which may cause backtracking")
        
        # Check for inefficient character classes
        if re.search(r'\[[^\]]{50,}\]', regex):
            warnings.append("Very large character class detected")
        
        # Check for unescaped special characters
        special_chars = r'.^$*+?{}[]|()'
        for char in special_chars:
            if char in regex and f'\\{char}' not in regex:
                # Check if it's intended
                if char == '.' and r'\.' not in regex:
                    warnings.append(f"Unescaped dot (.) will match any character")
                    break
        
        # Check complexity
        if len(regex) > 200:
            warnings.append("Pattern is very long and may be hard to maintain")
        
        return warnings
    
        
    def test_pattern(self, name: str, text: str) -> List[Dict[str, Any]]:
        """Test a specific pattern against text"""
        if name not in self.compiled_patterns:
            return []
        
        pattern = self.compiled_patterns[name]
        matches = []
        
        try:
            for match in pattern.finditer(text):
                captured_text = match.group(0)

                # Lógica para manejar tags unidos por slash
                if name == 'tag_compuesto_con_slash':
                    tags_a_procesar = captured_text.split('/')
                else:
                    # Evita que otros patrones procesen erróneamente un tag compuesto
                    if '/' in captured_text and len(captured_text.split('/')) == 2:
                        continue
                    tags_a_procesar = [captured_text]
                
                for individual_tag in tags_a_procesar:
                    try:
                        start_pos = text.find(individual_tag, match.start(), match.end())
                        if start_pos == -1: start_pos = match.start() # Fallback
                        end_pos = start_pos + len(individual_tag)
                    except:
                        start_pos, end_pos = match.start(), match.end()

                    matches.append({
                        'pattern_name': name,
                        'tag_capturado': individual_tag,
                        'start': start_pos,
                        'end': end_pos,
                        'context': self.get_match_context(text, start_pos, end_pos)
                    })
            
            self.pattern_stats[name]['matches'] += len(matches)
        except Exception as e:
            self.logger.error(f"Error testing pattern '{name}': {e}")
            self.pattern_stats[name]['errors'] += 1
        
        return matches
    
    def get_match_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Get context around a match"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        context = text[context_start:context_end]
        
        # Mark the match
        match_offset = start - context_start
        marked_context = (
            context[:match_offset] + 
            f"[{context[match_offset:match_offset + (end - start)]}]" + 
            context[match_offset + (end - start):]
        )
        
        return marked_context.replace('\n', ' ').strip()
    
        
    def find_all_matches(self, text: str) -> List[Dict[str, Any]]:
        """
        Encuentra todas las coincidencias y devuelve los detalles enriquecidos con la taxonomía.
        """
        all_matches = []
        # Podrías definir un orden de prioridad si es necesario
        for name, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                # Obtener la metadata de la taxonomía para este patrón
                taxonomy_info = self.taxonomy_data.get(name, {})
                
                match_details = {
                    'tag_capturado': match.group(0),
                    'pattern_name': name,
                    'start': match.start(),
                    'end': match.end(),
                    'context': self.get_match_context(text, match.start(), match.end()),
                    # --- ¡Enriquecer con datos de la taxonomía! ---
                    'clasificacion_level_1': taxonomy_info.get('level_1'),
                    'clasificacion_level_2': taxonomy_info.get('level_2'),
                    'tag_code': taxonomy_info.get('tag_code')
                }
                all_matches.append(match_details)
        
        # Lógica para resolver solapamientos (la más larga y específica gana)
        if not all_matches:
            return []
            
        all_matches.sort(key=lambda m: (m['start'], -len(m['tag_capturado'])))
        final_matches = []
        last_end_pos = -1
        for match in all_matches:
            if match['start'] >= last_end_pos:
                final_matches.append(match)
                last_end_pos = match['end']

        return final_matches
    
   
        
    def validate_custom_patterns(self, patterns_json: str) -> Dict[str, Any]:
        """
        Validate custom patterns from JSON.
        Esta es la versión CORREGIDA y AUTOCONTENIDA.
        """
        result = {
            'valid': True,
            'patterns': {},
            'error': None
        }
        
        try:
            patterns = json.loads(patterns_json)
            
            if not isinstance(patterns, dict):
                result['valid'] = False
                result['error'] = "Patterns must be a JSON object"
                return result
            
            for name, regex in patterns.items():
                # --- INICIO DE LA LÓGICA CORREGIDA ---
                # En lugar de llamar a una función de ayuda, hacemos la validación aquí mismo.
                pattern_result = {'valid': False, 'error': None}
                try:
                    re.compile(regex)
                    pattern_result['valid'] = True
                except re.error as e:
                    pattern_result['error'] = str(e)
                    result['valid'] = False # Si un patrón es inválido, todo el conjunto es inválido.
                
                result['patterns'][name] = pattern_result
                # --- FIN DE LA LÓGICA CORREGIDA ---
                    
        except json.JSONDecodeError as e:
            result['valid'] = False
            result['error'] = f"Invalid JSON: {e}"
            
        return result
    
    def export_patterns(self, include_stats: bool = True) -> Dict[str, Any]:
        """
        Export all patterns
        
        Args:
            include_stats: Include usage statistics
            
        Returns:
            Exported patterns data
        """
        export_data = {
            'patterns': {},
            'metadata': {
                'version': '1.0',
                'pattern_count': len(self.patterns)
            }
        }
        
        for name, pattern_info in self.patterns.items():
            pattern_data = {
                'regex': pattern_info['regex'],
                'category': pattern_info['category']
            }
            
            # Add description and examples from defaults
            if name in self.default_patterns:
                pattern_data['description'] = self.default_patterns[name].get('description', '')
                pattern_data['examples'] = self.default_patterns[name].get('examples', [])
            
            # Add statistics if requested
            if include_stats and name in self.pattern_stats:
                pattern_data['stats'] = dict(self.pattern_stats[name])
            
            export_data['patterns'][name] = pattern_data
        
        return export_data
    
    def import_patterns(self, patterns_data: Dict[str, Any], merge: bool = True) -> Dict[str, Any]:
        """
        Import patterns from data
        
        Args:
            patterns_data: Patterns data to import
            merge: Merge with existing patterns or replace
            
        Returns:
            Import results
        """
        result = {
            'imported': 0,
            'failed': 0,
            'errors': []
        }
        
        if not merge:
            # Clear existing custom patterns (keep defaults)
            self.patterns = {}
            self.compiled_patterns = {}
            self.load_default_patterns()
        
        patterns = patterns_data.get('patterns', {})
        
        for name, pattern_info in patterns.items():
            regex = pattern_info.get('regex', '')
            category = pattern_info.get('category', 'custom')
            
            if self.add_pattern(name, regex, category):
                result['imported'] += 1
            else:
                result['failed'] += 1
                result['errors'].append(f"Failed to import pattern '{name}'")
        
        return result
    
    def suggest_similar_patterns(self, tag: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Suggest patterns that might match similar tags
        
        Args:
            tag: Example tag
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar patterns
        """
        suggestions = []
        
        for name, pattern_info in self.patterns.items():
            pattern = self.compiled_patterns[name]
            
            # Generate example tags from pattern
            examples = []
            if name in self.default_patterns:
                examples = self.default_patterns[name].get('examples', [])
            
            # Check similarity
            for example in examples:
                if HAS_LEVENSHTEIN:
                    similarity = Levenshtein.ratio(tag.upper(), example.upper())
                else:
                    similarity = simple_similarity(tag, example)
                
                if similarity >= threshold:
                    suggestions.append({
                        'pattern_name': name,
                        'pattern_regex': pattern_info['regex'],
                        'example': example,
                        'similarity': round(similarity, 2),
                        'category': pattern_info['category']
                    })
                    break
        
        # Sort by similarity
        suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        
        return suggestions
    
    def generate_pattern_from_examples(self, examples: List[str]) -> Optional[str]:
        """
        Attempt to generate a pattern from examples
        
        Args:
            examples: List of example tags
            
        Returns:
            Generated regex pattern or None
        """
        if not examples:
            return None
        
        try:
            # Find common structure
            # This is a simplified version - could be enhanced with more sophisticated analysis
            
            # Check if all examples have same length
            lengths = [len(ex) for ex in examples]
            if len(set(lengths)) == 1:
                # Same length - analyze character by character
                pattern_parts = []
                
                for i in range(lengths[0]):
                    chars = [ex[i] for ex in examples]
                    
                    if all(c.isdigit() for c in chars):
                        pattern_parts.append(r'\d')
                    elif all(c.isalpha() and c.isupper() for c in chars):
                        pattern_parts.append('[A-Z]')
                    elif all(c == chars[0] for c in chars):
                        # All same character
                        pattern_parts.append(re.escape(chars[0]))
                    else:
                        # Mixed - use character class
                        unique_chars = ''.join(sorted(set(chars)))
                        pattern_parts.append(f'[{re.escape(unique_chars)}]')
                
                return r'\b' + ''.join(pattern_parts) + r'\b'
            
            else:
                # Different lengths - try to find common prefix/suffix
                # This is a placeholder for more complex logic
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating pattern: {e}")
            return None
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all patterns"""
        stats = {
            'total_patterns': len(self.patterns),
            'categories': defaultdict(int),
            'most_used': [],
            'least_used': [],
            'error_prone': []
        }
        
        # Count by category
        for pattern_info in self.patterns.values():
            stats['categories'][pattern_info['category']] += 1
        
        # Sort by usage
        pattern_usage = [
            (name, data['matches']) 
            for name, data in self.pattern_stats.items()
        ]
        pattern_usage.sort(key=lambda x: x[1], reverse=True)
        
        # Most and least used
        if pattern_usage:
            stats['most_used'] = pattern_usage[:5]
            stats['least_used'] = pattern_usage[-5:]
        
        # Error prone patterns
        error_patterns = [
            (name, data['errors']) 
            for name, data in self.pattern_stats.items() 
            if data['errors'] > 0
        ]
        error_patterns.sort(key=lambda x: x[1], reverse=True)
        stats['error_prone'] = error_patterns[:5]
        
        return stats
