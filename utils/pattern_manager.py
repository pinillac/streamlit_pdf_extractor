"""
Pattern Manager Utility
Manages regex patterns for equipment tag extraction
"""
import re
import json
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import logging

class PatternManager:
    def __init__(self, taxonomy_file='taxonomy_1.csv'):
        self.logger = logging.getLogger(__name__)
        self.compiled_patterns = {} 
        self.taxonomy_data = {}
        self.load_patterns_from_csv(taxonomy_file)

    def load_patterns_from_csv(self, taxonomy_file: str):
        try:
            taxonomy_path = Path(__file__).parent.parent / taxonomy_file
            df = pd.read_csv(taxonomy_path, encoding='utf-8', dtype=str, engine='python')
            df.fillna('', inplace=True)
            df.columns = df.columns.str.strip()
            df['pattern_name'] = df['Level 1'] + '_' + df['Level 2']

            for index, row in df.iterrows():
                if not row['Regex Pattern']: continue
                try:
                    self.compiled_patterns[row['pattern_name']] = re.compile(row['Regex Pattern'])
                    self.taxonomy_data[row['pattern_name']] = row.to_dict()
                except re.error as e:
                    self.logger.error(f"Error compilando regex para '{row['pattern_name']}': {row['Regex Pattern']} -> {e}")
            self.logger.info(f"Cargados {len(self.compiled_patterns)} patrones desde {taxonomy_file}")
        except Exception as e:
            self.logger.error(f"Error crítico procesando el archivo de taxonomía: {e}")
    
    def get_match_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end].replace('\n', ' ').strip()
        
    def find_all_matches(self, text: str) -> List[Dict[str, Any]]:
        all_matches = []
        priority_order = [
            'Piping_P&ID Line Number',
            'Instrument_Standard With-Hyphen', 'Instrument_Standard No-Hyphen',
            'Equipment_Standard With-Hyphen', 'Equipment_Standard No-Hyphen',
            'ElectricalEquipment_Bus-Based', 'ElectricalEquipment_Unit-Based',
            'Structure_Steel No-Hyphen', 'Structure_Steel With-Hyphen',
            'Document_Standard',
        ]
        
        all_pattern_names = list(self.compiled_patterns.keys())
        ordered_patterns = [p for p in priority_order if p in all_pattern_names]
        ordered_patterns.extend([p for p in all_pattern_names if p not in ordered_patterns])

        for name in ordered_patterns:
            pattern = self.compiled_patterns[name]
            for match in pattern.finditer(text):
                taxonomy_info = self.taxonomy_data.get(name, {})
                match_details = {
                    'tag_capturado': match.group(0),
                    'pattern_name': name,
                    'start': match.start(), 'end': match.end(),
                    'context': self.get_match_context(text, match.start(), match.end()),
                    'clasificacion_level_1': taxonomy_info.get('Level 1'),
                    'clasificacion_level_2': taxonomy_info.get('Level 2'),
                    'tag_code': taxonomy_info.get('Tag Code'),
                    'match_object': match
                }
                all_matches.append(match_details)
        
        if not all_matches: return []
        
        all_matches.sort(key=lambda m: (m['start'], -len(m['tag_capturado'])))
        final_matches, last_end_pos = [], -1
        for match in all_matches:
            if match['start'] >= last_end_pos:
                final_matches.append(match)
                last_end_pos = match['end']
        return final_matches