"""
Pattern Manager Utility
Manages regex patterns for equipment tag extraction
"""

import re
import json
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
    """Manages and validates regex patterns for tag extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = {}
        self.compiled_patterns = {}
        self.load_default_patterns()
        
    def load_default_patterns(self):
        """
        Load a comprehensive, categorized set of robust regex patterns.
        Each pattern is designed for a specific tag family.
        """
        self.default_patterns = {
            # --- FAMILIA: INSTRUMENTOS (Con y Sin Unidad) ---
            "instrumento_con_unidad": {
                "regex": r"(?<!\d)(\d{3})-([A-Z]{1,4}-\d{3,5}[A-Z]?)(?![A-Z0-9-])",
                "description": "Instrumento con prefijo de unidad (ej: 010-AI-0002)",
                "category": "Instrumento"
            },
            "instrumento_sin_unidad": {
                "regex": r"(?<![A-Z0-9-])([A-Z]{1,4}-\d{3,5}[A-Z]?)(?![A-Z0-9-])",
                "description": "Instrumento sin prefijo de unidad (ej: TG-0746)",
                "category": "Instrumento"
            },

            # --- FAMILIA: EQUIPOS (Con y Sin Unidad) ---
            "equipo_con_unidad_sin_guion": {
                "regex": r"(?<![A-Za-z0-9])(\d{3})([A-Z]{1,4})(\d{3,5}[A-Z]?)(?![A-Za-z0-9])",
                "description": "Equipo con unidad sin guion (ej: 011C0001, 018P2001A)",
                "category": "Equipo"
            },
            
            # --- FAMILIA: ELÉCTRICO ---
            "componente_electrico_comun": {
                "regex": r"(?<![A-Za-z0-9])(BUS|LC|UPS|MCC|SWG|XFMR)(\d+[A-Z]?)(?![A-Za-z0-9])",
                "description": "Componentes eléctricos comunes (ej: BUS4328N)",
                "category": "Eléctrico"
            },

            # --- FAMILIA: DOCUMENTOS ---
            "documento_estandar": {
                "regex": r"\b(\d{3}-[A-Z]-[A-Z]{3}-\d{5}-\d{3})\b",
                "description": "Documento con formato estándar",
                "category": "Documento"
            },

            # --- FAMILIA: TUBERÍAS (PIPING) ---
            "tuberia_compleja": {
                "regex": r"\b((?:\d{3}-)?\d{1,2}\"-[A-Z]{2,4}-\d{4,6}-[A-Z0-9-]+)\b",
                "description": "Piping Line Number (Complex Format)",
                "category": "Piping"
            },

            # --- FAMILIA: EDIFICIOS / ESTRUCTURAS ---
            "estructura_con_unidad": {
                "regex": r"(?<![A-Za-z0-9])(\d{3})(STR|PR|PLF|SSTW|SSTR|SLT)(\d+)(?![A-Za-z0-9])",
                "description": "Estructuras con prefijo de unidad (ej: 051STR100)",
                "category": "Edificio"
            },
        }
        
        for name, pattern_info in self.default_patterns.items():
            self.add_pattern(name, pattern_info['regex'], pattern_info.get('category'))

    def add_pattern(self, name: str, regex: str, category: str = 'custom') -> bool:
        try:
            self.patterns[name] = {'regex': regex, 'category': category, 'compiled': re.compile(regex)}
            self.compiled_patterns[name] = self.patterns[name]['compiled']
            return True
        except re.error as e:
            self.logger.error(f"Invalid regex pattern '{name}': {e}")
            return False

    def find_all_matches(self, text: str) -> List[Dict[str, Any]]:
        """Finds all matches, handling overlaps by prioritizing longer, more specific matches."""
        all_matches = []
        # Ordenar patrones por longitud de regex para dar prioridad a los más específicos
        sorted_patterns = sorted(self.patterns.items(), key=lambda item: len(item[1]['regex']), reverse=True)

        for name, pattern_info in sorted_patterns:
            pattern = pattern_info['compiled']
            for match in pattern.finditer(text):
                all_matches.append({
                    'pattern_name': name,
                    'match_obj': match,
                    'tag_capturado': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'context': self.get_match_context(text, match.start(), match.end())
                })
        
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

    def get_match_context(self, text: str, start: int, end: int, context_size: int = 40) -> str:
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end].replace('\n', ' ').strip()