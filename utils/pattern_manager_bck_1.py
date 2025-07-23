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
    """Manages and validates regex patterns for tag extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = {}
        self.compiled_patterns = {}
        self.pattern_stats = defaultdict(lambda: {'matches': 0, 'errors': 0})
        self.load_default_patterns()
        
    def load_default_patterns(self):
        """Load default equipment patterns"""
        self.default_patterns = {
            "annunciator": {
              "regex": "^ \\d{2,3}ANN\\d{4}[A-Z]\\d$",
              "description": "Annunciator",
              "category": "Electrical Equipment",
              "examples": ["ANN1234A5", "ANN5678B9"],
              "isa_category": "ANN"            
            },
            "battery_bank": {
              "regex": "^ \\d{2,3}BAT\\d{3}[A-Z]\\d$",
              "description": "Battery Bank",
              "category": "Electrical Equipment",
              "examples": ["BAT123A4", "BAT567B8"],
              "isa_category": "BAT"  
            },
            "breaker": {
              "regex": "^ \\d{2,3}BK[A-Z]{2}\\d{2}$",
              "description": "Breaker",
              "category": "Electrical Equipment",
              "examples": ["BKAP01", "BKDC12"],
              "isa_category": "BK"
              
            },
            "bus_duct": {
              "regex": "^ \\d{2,3}BD\\d+$",
              "description": "Bus Duct",
              "category": "Electrical Equipment",
              "examples": ["BD1", "BD15"],
              "isa_category": "BD"
              
            },
            "cathodic_protection": {
              "regex": "^ \\d{2,3}CP[A-Z]+\\d+$",
              "description": "Cathodic Protection",
              "category": "Electrical Equipment",
              "examples": ["CPIMP1", "CPTEST4"],
              "isa_category": "CP"              
            },
            "cable_tray": {
              "regex": "^ \\d{2,3}CT[A-Z]+\\d+[A-Z]?$",
              "description": "Cable Tray",
              "category": "Electrical Equipment",
              "examples": ["CTI1", "CTP2A"],
              "isa_category": "CT"
              
            },
            "diesel_generator": {
              "regex": "^ \\d{2,3}DG\\d$",
              "description": "Diesel Generator",
              "category": "Electrical Equipment",
              "examples": ["DG1", "DG2"],
              "isa_category": "DG"
              
            },
            "emergency_generator": {
              "regex": "^ \\d{2,3}EG\\d$",
              "description": "Emergency Generator",
              "category": "Electrical Equipment",
              "examples": ["EG1", "EG2"],
              "isa_category": "EG"
              
            },
            "electric_heat_trace": {
              "regex": "^ \\d{2,3}ETP\\d+C\\d+$",
              "description": "Electric Heat Trace",
              "category": "Electrical Equipment",
              "examples": ["ETP1C1", "ETP2C12"],
              "isa_category": "ET"
              
            },
            "junction_box": {
              "regex": "^ \\d{2,3}JB\\d{4}$",
              "description": "Junction Box",
              "category": "Electrical Equipment",
              "examples": ["JB1001", "JB2034"],
              "isa_category": "JB"
              
            },
            "lighting_fixture": {
              "regex": "^ \\d{2,3}LF\\d{4}$",
              "description": "Lighting Fixture",
              "category": "Electrical Equipment",
              "examples": ["LF1001", "LF2034"],
              "isa_category": "LF"
              
            },
            "lighting_panel": {
              "regex": "^ \\d{2,3}LP[A-Z0-9]+[A-Z]?$",
              "description": "Lighting Panel",
              "category": "Electrical Equipment",
              "examples": ["LPA", "LP1B"],
              "isa_category": "LP"
              
            },
            "motor_control_center": {
              "regex": "^ \\d{2,3}MCC\\d+[A-Z]$",
              "description": "Motor Control Center",
              "category": "Electrical Equipment",
              "examples": ["MCC1A", "MCC2B"],
              "isa_category": "MCC"
              
            },
            "motor": {
              "regex": "^ \\d{2,3}MTR\\d{4}$",
              "description": "Motor",
              "category": "Electrical Equipment",
              "examples": ["MTR1234", "MTR5678"],
              "isa_category": "MTR"
              
            },
            "receptacle": {
              "regex": "^ \\d{2,3}RCPT\\d{4}$",
              "description": "Receptacle",
              "category": "Electrical Equipment",
              "examples": ["RCPT1001", "RCPT2034"],
              "isa_category": "RCPT"
              
            },
            "switch": {
              "regex": "^ \\d{2,3}SW\\d{4}$",
              "description": "Switch",
              "category": "Electrical Equipment",
              "examples": ["SW1001", "SW2034"],
              "isa_category": "SW"
              
            },
            "switchgear": {
              "regex": "^ \\d{2,3}SWG\\d{3}[A-Z]?$",
              "description": "Switchgear",
              "category": "Electrical Equipment",
              "examples": ["SWG001", "SWG002A"],
              "isa_category": "SWG"
              
            },
            "uninterruptible_power_supply": {
              "regex": "^ \\d{2,3}UPS\\d{3}$",
              "description": "Uninterruptible Power Supply",
              "category": "Electrical Equipment",
              "examples": ["UPS001", "UPS002"],
              "isa_category": "UPS"
              
            },
            "transformer": {
              "regex": "^ \\d{2,3}XFMR\\d{3}[A-Z]?$",
              "description": "Transformer",
              "category": "Electrical Equipment",
              "examples": ["XFMR001", "XFMR002B"],
              "isa_category": "XFMR"
              
            },
            "air_conditioning": {
              "regex": "^ \\d{2,3}AC\\d{3,4}[A-Z]?$",
              "description": "Air Conditioning",
              "category": "HVAC",
              "examples": ["10AC123", "123AC1234A"],
              "isa_category": "AC"
              
            },
            "air_handling_unit": {
              "regex": "^ \\d{2,3}AHU\\d{3,4}[A-Z]?$",
              "description": "Air Handling Unit",
              "category": "HVAC",
              "examples": ["10AHU123", "123AHU1234A"],
              "isa_category": "AHU"
              
            },
            "chiller": {
              "regex": "^ \\d{2,3}CH\\d{3}[A-Z]?$",
              "description": "Chiller",
              "category": "HVAC",
              "examples": ["10CH001", "123CH002A"],
              "isa_category": "CH"
              
            },
            "damper": {
              "regex": "^ \\d{2,3}DMP\\d{4}$",
              "description": "Damper",
              "category": "HVAC",
              "examples": ["10DMP1001", "123DMP2002"],
              "isa_category": "DMP"
              
            },
            "exhaust_fan": {
              "regex": "^ \\d{2,3}EF\\d{3,4}[A-Z]?$",
              "description": "Exhaust Fan",
              "category": "HVAC",
              "examples": ["10EF123", "123EF1234A"],
              "isa_category": "EF"
              
            },
            "compressor_blower": {
              "regex": "^ \\d{2,3}K\\d{3,4}[A-Z]?$",
              "description": "Compressor / Blower",
              "category": "Mechanical",
              "examples": ["10K123", "123K1234A"],
              "isa_category": "K"
              
            },
            "pump": {
                "regex": r"(?<!\d)(\d{3})[-]?([P])[-]?(\d{3,4})[-]?([A-Z](?:/[A-Z])?)?(?![A-Z0-9])",
                "description": "Bomba (Pump) con múltiples formatos, incluyendo A/B dual y separadores opcionales.",
                "category": "Equipo",
                "examples": ["123P1234A", "123P1234A/B", "123-P-1234-A/B", "010P0001A",  "010-P-0001-A" ],
                "isa_category": "P"
            },
            "tank_vessel": {
              "regex": "^ \\d{2,3}TK\\d{3,4}[A-Z]?$",
              "description": "Tank / Vessel",
              "category": "Mechanical",
              "examples": ["10TK123", "123TK1234A"],
              "isa_category": "TK"
              
            },
            "analyzer": {
              "regex": "^ \\d{2,3}\\d+-A[A-Z]*\\d+-\\d+$",
              "description": "Analyzer",
              "category": "Instrument",
              "examples": ["100-AE101-1", "200-AT202-2"],
              "isa_category": "A"
              
            },
            "burner": {
              "regex": "^ \\d+-B[A-Z]*\\d+-\\d+$",
              "description": "Burner",
              "category": "Instrument",
              "examples": ["100-BE101-1", "200-BT202-2"],
              "isa_category": "B"
              
            },
            "control_valve": {
              "regex": "^ \\d+-C[A-Z]*\\d+-\\d+$",
              "description": "Control Valve",
              "category": "Instrument",
              "examples": ["100-CV101-1", "200-FCV202-2"],
              "isa_category": "C"
              
            },
            "damper_louver": {
              "regex": "^ \\d+-D[A-Z]*\\d+-\\d+$",
              "description": "Damper / Louver",
              "category": "Instrument",
              "examples": ["100-DE101-1", "200-DT202-2"],
              "isa_category": "D"
              
            },
            "sensor": {
              "regex": "^ \\d+-E[A-Z]*\\d+-\\d+$",
              "description": "Sensor",
              "category": "Instrument",
              "examples": ["100-LE101-1", "200-TE202-2"],
              "isa_category": "E"
              
            },
            "flow": {
              "regex": "^ \\d+-F[A-Z]*\\d+-\\d+$",
              "description": "Flow",
              "category": "Instrument",
              "examples": ["100-FE101-1", "200-FT202-2"],
              "isa_category": "F"
              
            },
            "gauge": {
              "regex": "^ \\d+-G[A-Z]*\\d+-\\d+$",
              "description": "Gauge",
              "category": "Instrument",
              "examples": ["100-PG101-1", "200-TG202-2"],
              "isa_category": "G"
              
            },
            "hand": {
              "regex": "^ \\d+-H[A-Z]*\\d+-\\d+$",
              "description": "Hand",
              "category": "Instrument",
              "examples": ["100-HC101-1", "200-HS202-2"],
              "isa_category": "H"
              
            },
            "current": {
              "regex": "^ \\d+-I[A-Z]*\\d+-\\d+$",
              "description": "Current",
              "category": "Instrument",
              "examples": ["100-IE101-1", "200-IT202-2"],
              "isa_category": "I"
              
            },
            "power": {
              "regex": "^ \\d+-J[A-Z]*\\d+-\\d+$",
              "description": "Power",
              "category": "Instrument",
              "examples": ["100-JE101-1", "200-JT202-2"],
              "isa_category": "J"
              
            },
            "time": {
              "regex": "^ \\d+-K[A-Z]*\\d+-\\d+$",
              "description": "Time",
              "category": "Instrument",
              "examples": ["100-KC101-1", "200-KT202-2"],
              "isa_category": "K"
              
            },
            "level": {
              "regex": "^ \\d+-L[A-Z]*\\d+-\\d+$",
              "description": "Level",
              "category": "Instrument",
              "examples": ["100-LE101-1", "200-LT202-2"],
              "isa_category": "L"
              
            },
            "moisture": {
              "regex": "^ \\d+-M[A-Z]*\\d+-\\d+$",
              "description": "Moisture",
              "category": "Instrument",
              "examples": ["100-ME101-1", "200-MT202-2"],
              "isa_category": "M"
              
            },
            "orifice": {
              "regex": "^ \\d+-O[A-Z]*\\d+-\\d+$",
              "description": "Orifice",
              "category": "Instrument",
              "examples": ["100-FO101-1", "200-PO202-2"],
              "isa_category": "O"
              
            },
            "pressure": {
              "regex": "^ \\d+-P[A-Z]*\\d+-\\d+$",
              "description": "Pressure",
              "category": "Instrument",
              "examples": ["100-PE101-1", "200-PT202-2"],
              "isa_category": "P"
              
            },
            "quantity": {
              "regex": "^ \\d+-Q[A-Z]*\\d+-\\d+$",
              "description": "Quantity",
              "category": "Instrument",
              "examples": ["100-QE101-1", "200-QT202-2"],
              "isa_category": "Q"
              
            },
            "radiation": {
              "regex": "^ \\d+-R[A-Z]*\\d+-\\d+$",
              "description": "Radiation",
              "category": "Instrument",
              "examples": ["100-RE101-1", "200-RT202-2"],
              "isa_category": "R"
              
            },
            "speed": {
              "regex": "^ \\d+-S[A-Z]*\\d+-\\d+$",
              "description": "Speed",
              "category": "Instrument",
              "examples": ["100-SE101-1", "200-ST202-2"],
              "isa_category": "S"
              
            },
            "temperature": {
              "regex": "^ \\d+-T[A-Z]*\\d+-\\d+$",
              "description": "Temperature",
              "category": "Instrument",
              "examples": ["100-TE101-1", "200-TT202-2"],
              "isa_category": "T"
              
            },
            "multivariable": {
              "regex": "^ \\d+-U[A-Z]*\\d+-\\d+$",
              "description": "Multivariable",
              "category": "Instrument",
              "examples": ["100-UE101-1", "200-UT202-2"],
              "isa_category": "U"
              
            },
            "vibration": {
              "regex": "^ \\d+-V[A-Z]*\\d+-\\d+$",
              "description": "Vibration",
              "category": "Instrument",
              "examples": ["100-VE101-1", "200-VT202-2"],
              "isa_category": "V"
              
            },
            "well": {
              "regex": "^ \\d+-TW[A-Z]*\\d+-\\d+$",
              "description": "Well",
              "category": "Instrument",
              "examples": ["100-TW101-1", "200-TW202-2"],
              "isa_category": "W"
              
            },
            "building": {
                "regex": "^ \\d+?BLD-[A-Z]{2}\\d{2}$",
                "description": "Building",
                "category": "Building",
                "examples": ["BLD-AB01", "BLD-CD12"],
                "isa_category": "BLDG"
                
            },
            "room": {
                "regex": "^ \\d+?ROOM-[A-Z]{2}\\d{2}$",
                "description": "Room",
                "category": "Building",
                "examples": ["ROOM-AB01", "ROOM-CD12"],
                "isa_category": "ROOM"
                
            },
            "area": {
                "regex": "^ \\d+?AREA-\\d{2}$",
                "description": "Area",
                "category": "Building",
                "examples": ["AREA-01", "AREA-12"],
                "isa_category": "AREA"
                
            },
            "facility": {
                "regex": "^ \\d+?FAC-[A-Z]{2}\\d{2}$",
                "description": "Facility",
                "category": "Building",
                "examples": ["FAC-AB01", "FAC-CD12"],
                "isa_category": "FAC"
                
            },
            "floor": {
                "regex": "^ \\d+?FLR-\\d{2}$",
                "description": "Floor",
                "category": "Building",
                "examples": ["FLR-01", "FLR-12"],
                "isa_category": "FLR"
                
            },
            "yard": {
                "regex": "^ \\d+?YD-[A-Z]{2}\\d{2}$",
                "description": "Yard",
                "category": "Building",
                "examples": ["YD-AB01", "YD-CD12"],
                "isa_category": "YD"
            },
            "unidad_tipo_num_sin_separador": {
                 "regex": r"(?<![A-Za-z0-9])(\d{3})([A-Z]{1,4})(\d+[A-Z]?)(?![A-Za-z0-9])",
                 "description": "Tag con formato UNIDAD-TIPO-NUM sin separadores (e.g., 018M6018)",
                 "category": "compuesto",
                 "examples": ["018M6018", "051STR100", "051STR100", "153W1153", "110P0002A"],
                 "isa_category": "COMPOUND"
             },
            "instrumento_con_unidad": {
                "regex": r"(?<![A-Za-z0-9-])(\d{3})-([A-Z]{2,4})-(\d{3,4})([A-Z0-9]{1,2})?(?![A-Za-z0-9-])",
                "description": "Instrumento con prefijo de unidad de 3 dígitos (e.g., 180-TT-4004)",
                "category": "compuesto_instrumento", # Categoría más específica para priorizar
                "examples": ["180-ZSL-0001", "180-ZSL-0002B", "180-ZSL-0022"],
                "isa_category": "COMPOUND-INST"
            },
            # "instrumento_con_unidad_1": {
                # "regex": r"\b(\d{3})-([A-Z]{3,4})-(\d{3,5})([A-Z])?\b",
                # "description": "Instrumento con prefijo de unidad de 3 dígitos (e.g., 180-TT-4004)",
                # "category": "compuesto_instrumento", # Categoría más específica para priorizar
                # "examples": ["180-ZSL-0001", "180-ZSL-0002B", "180-ZSL-0022"],
                # "isa_category": "COMPOUND-INST_1"
            # },
             "componente_electrico_comun": {
                 #"regex": r"\b(BUS|LC|UPS)(\d+[A-Z]?)\b",
                 "regex": r"(?<![A-Za-z0-9])(BUS|LC|UPS)(\d+[A-Z]?)(?![A-Za-z0-9])",
                 "description": "Componentes eléctricos comunes con prefijo (BUS, LC, UPS)",
                 "category": "electrico",
                 "examples": ["BUS4328N", "LC451M5", "UPS018P0"],
                 "isa_category": "ELEC-COMMON"
             },
            # "bomba_classic": {
                # "regex": r"\b\d{3}P\d{4}(?:[A-Z](?:/[A-Z])?)?\b",
                # "description": "Classic Pump tags (e.g., 123P4567, 123P4567A)",
                # "examples": ["123P4567", "456P7890A", "789P0123B/C"],
                # "category": "equipment"
            # },
            # "horno_classic": {
                # "regex": r"\b\d{3}F\d{4}(?:[A-Z](?:/[A-Z])?)?\b",
                # "description": "Classic Furnace tags (e.g., 100F2500)",
                # "examples": ["100F2500", "205F3678A", "304F5555B/D"],
                # "category": "equipment"
            # },
            # "intercambiador_classic": {
                # "regex": r"\b\d{3}E\d{4}[A-Z]?\b",
                # "description": "Classic Heat exchanger tags",
                # "examples": ["123E4567", "456E7890A", "001E2345Z"],
                # "category": "equipment"
            # },
            # "recipiente_classic": {
                # "regex": r"\b\d{3}V\d{4}[A-Z]?\b",
                # "description": "Classic Vessel tags",
                # "examples": ["123V4567", "789V0123A", "555V9999Z"],
                # "category": "equipment"
            # },
            # "compresor_classic": {
                # "regex": r"\b\d{3}C\d{4}[A-Z]?\b",
                # "description": "Classic Compressor tags",
                # "examples": ["123C4567", "456C7890B", "999C1111A"],
                # "category": "equipment"
            # },
            # "instrumento_classic": {
                # "regex": r"\b[A-Z]{2,4}-\d{3,5}[A-Z]?\b",
                # "description": "Classic Instrument tags (ISA standard)",
                # "examples": ["PT-123", "FIC-12345A", "TRC-4567B", "LAHH-999"],
                # "category": "instrumentation"
            # },
            # "tuberia_classic": {
                # "regex": r"\b\d{1,2}\"-[A-Z]{2,4}-\d{4,6}-[A-Z0-9-]{1,20}\b",
                # "description": "Classic Piping system tags",
                # "examples": ["6\"-PT-1234-STEAM", "12\"-FW-123456-HOT-WATER"],
                # "category": "piping"
            # }
            # "bomba_classic": {
                # "regex": r"(?<![A-Z0-9])\d{3}P\d{4}(?:[A-Z](?:/[A-Z])?)?(?![A-Z0-9])",
                # "description": "Classic Pump tags (e.g., 123P4567, 123P4567A)",
                # "examples": ["123P4567", "456P7890A", "789P0123B/C"],
                # "category": "equipment"
            # },
            "horno_classic": {
                "regex": r"(?<![A-Z0-9])\d{3}F\d{4}(?:[A-Z](?:/[A-Z])?)?(?![A-Z0-9])",
                "description": "Classic Furnace tags (e.g., 100F2500)",
                "examples": ["100F2500", "205F3678A", "304F5555B/D"],
                "category": "equipment"
            },
            "intercambiador_classic": {
                "regex": r"(?<![A-Z0-9])\d{3}E\d{4}[A-Z]?(?![A-Z0-9])",
                "description": "Classic Heat exchanger tags",
                "examples": ["123E4567", "456E7890A", "001E2345Z"],
                "category": "equipment"
            },
            "recipiente_classic": {
                "regex": r"(?<![A-Z0-9])\d{3}V\d{4}[A-Z]?(?![A-Z0-9])",
                "description": "Classic Vessel tags",
                "examples": ["123V4567", "789V0123A", "555V9999Z"],
                "category": "equipment"
            },
            "compresor_classic": {
                "regex": r"(?<![A-Z0-9])\d{3}C\d{4}[A-Z]?(?![A-Z0-9])",
                "description": "Classic Compressor tags",
                "examples": ["123C4567", "456C7890B", "999C1111A"],
                "category": "equipment"
            },
             "Safety Shower": {
                "regex": r"(?<![A-Z0-9])\d{3}SSE\d{4}[A-Z]?(?![A-Z0-9])",
                "description": "Safety Shower tags",
                "examples": ["123C4567", "456C7890B", "999C1111A"],
                "category": "equipment"
            },
            "instrumento_classic": {
                "regex": r'(?<![A-Z0-9-])(PT|FT|TE|LT|LIC|PI|FIC|TG|PIT|HS|ZSL|XS|XHS|XI)-\d{3,5}[A-Z]?(?![A-Z0-9-])',
                "description": "Classic Instrument tags (ISA standard)",
                "examples": ["PT-123", "FIC-12345A", "TRC-4567B", "LAHH-999"],
                "category": "instrumentation"
            },
            "Columna": {
                "regex": r'\b\d{3}C\d{4}[A-Z]?\b',
                "description": "Column Equipment Tag (e.g., 018C0001, 100P2034A)",
                "category": "equipment",  
                "examples": ["018C0001", "123P4567A", "400V1001"],
                "isa_category": "C"
            },
            "tuberia_compleja": {
                "regex": r'\b(?:\d{3}-)?\d{1,2}"-[A-Z]{2,4}-\d{4,6}-[A-Z0-9]+(?:-[a-zA-Z]+)?\b',
                "description": "Piping Line Number (Complex Format)",
                "category": "piping",
                "examples": ["018-4\"-BFW-7008-D3S-lh", "12\"-CW-12345-A1B"],
                "isa_category": "PIPE" # O lo que corresponda
            },
            # --- PEGA AQUÍ LOS NUEVOS PATRONES ---
            "documento_estandar": {
                "regex": r'\b\d{3}-[A-Z]-[A-Z]{3}-\d{5}-\d{3}\b',
                "description": "Documento con formato estandar (Area-Disciplina-Tipo-Sec-Rev)",
                "category": "documento",
                "examples": ["010-P-CRD-00001-001", "050-A-PID-55001-001"],
                "isa_category": "DOC"
            },
            "documento_proyecto_largo": {
                "regex": r'\b\d{6}[A-Z]-\d{2,4}[A-Z]?-[A-Z]{2,3}(?:-\d+)+(?:-\d+)?\b',
                "description": "Documento de proyecto con formato largo y múltiples segmentos",
                "category": "documento",
                "examples": ["076551C-079-SP-0270-001", "076551C-021-PFD-0010-001-001"],
                "isa_category": "DOC-PROJ"
            },
            "documento_proyecto_corto": {
                "regex": r'\b\d{6}[A-Z]-\d+[A-Z]?-[A-Z]{2,3}-\d+\b',
                "description": "Documento de proyecto con formato corto",
                "category": "documento",
                "examples": ["076551C-021-ML-214", "076551C-000D-RR-1044"],
                "isa_category": "DOC-PROJ-S"
            },
            "sample_probe_tag": {
                "regex": r'\b\d{3}-AP-\d{4}\b',
                "description": "Sample Probe Tag (e.g., 077-AP-1001)",
                "category": "instrumento",
                "examples": ["077-AP-1001"],
                "isa_category": "PROBE"
            },
            "analyser_tag": {
                "regex": r'\b\d{3}-A[A-Z]{2}-\d{4}\b',
                "description": "Analyser Tag (ISA Standard, e.g., 077-AIT-1001)",
                "category": "instrumento",
                "examples": ["077-AIT-1001", "077-AE-1002"],
                "isa_category": "ANALYSER"
            },
            "Pressure Diferential indicator": {
                "regex": r'\b\d{3}-PD[A-Z]{2}-\d{4}\b',
                "description": "Pressure Diferential indicator",
                "category": "instrumento",
                "examples": ["077-PDI-1001", "077-PDIT-1002"],
                "isa_category": "Pressure"
            },
            "tag_compuesto_con_slash": {
               #"regex": r"(?<![A-Za-z0-9])(\d{3}[A-Z]\d{4}[A-Z]?/\d{3}[A-Z]\d{4}[A-Z]?)(?![A-Za-z0-9])",
               "regex": r"(?<![A-Za-z0-9])(\d{3}[A-Z]{1,4}\d+[A-Z]?/\d{3}[A-Z]{1,4}\d+[A-Z]?)(?![A-Za-z0-9])",
               "description": "Pares de tags unidos por un slash (e.g., 018T1001A/018T2001A)",
               "category": "compuesto",
               "examples": ["018T1001A/018T2001A"],
               "isa_category": "DUAL-SLASH"
            }
 }       
        # Compile default patterns
        for name, pattern_info in self.default_patterns.items():
            self.add_pattern(name, pattern_info['regex'], pattern_info.get('category', 'custom'))
    
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
    
    # def test_pattern(self, name: str, text: str) -> List[Dict[str, Any]]:
        # """
        # Test a specific pattern against text
        
        # Args:
            # name: Pattern name
            # text: Text to search
            
        # Returns:
            # List of matches
        # """
        # if name not in self.compiled_patterns:
            # return []
        
        # pattern = self.compiled_patterns[name]
        # matches = []
        
        # try:
            # for match in pattern.finditer(text):
                # matches.append({
                    # 'pattern': name,
                    # 'text': match.group(),
                    # 'start': match.start(),
                    # 'end': match.end(),
                    # 'context': self.get_match_context(text, match.start(), match.end())
                # })
            
            # # Update statistics
            # self.pattern_stats[name]['matches'] += len(matches)
            
        # except Exception as e:
            # self.logger.error(f"Error testing pattern '{name}': {e}")
            # self.pattern_stats[name]['errors'] += 1
        
        # return matches
        
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
    
    # def find_all_matches(self, text: str, categories: List[str] = None) -> List[Dict[str, Any]]:
        # """
        # Find all matches using all patterns
        
        # Args:
            # text: Text to search
            # categories: Optional list of categories to filter
            
        # Returns:
            # List of all matches
        # """
        # all_matches = []
        
        # for name, pattern_info in self.patterns.items():
            # # Filter by category if specified
            # if categories and pattern_info['category'] not in categories:
                # continue
            
            # matches = self.test_pattern(name, text)
            # all_matches.extend(matches)
        
        # # Sort by position
        # all_matches.sort(key=lambda x: x['start'])
        
        # return all_matches
        
    def find_all_matches(self, text: str, categories: List[str] = None) -> List[Dict[str, Any]]:
        """Find all matches using all patterns, handling overlaps and ensuring the best match is kept."""
        all_matches = []
        
        # Orden de prioridad explícito: de más específico a más general
        pattern_order = [
            # --- Patrones Compuestos y de Alta Especificidad ---
            "tag_compuesto_con_slash",
            "documento_proyecto_largo",
            "documento_estandar",
            "documento_proyecto_corto",
            "tuberia_compleja",
            "instrumento_con_unidad",
            #"instrumento_con_unidad_1",
            "analyser_tag",
            "sample_probe_tag",
            "Safety Shower",
            "Pressure Diferential indicator",
            
            # --- Patrones de Equipo con Unidad (sin guiones) ---
            "unidad_tipo_num_sin_separador",
            
            # --- Patrones de Equipos Específicos (por prefijo) ---
            "pump",
            "horno_classic",
            "intercambiador_classic",
            "recipiente_classic",
            "Columna",              # Compite con 'compresor_classic' por la 'C'
            "compresor_classic",
            "motor_control_center",
            "switchgear",
            "transformer",
            "electric_heat_trace",
            
            # --- Patrones de Instrumentos (sin unidad) ---
            "instrumento_classic",  # Captura grupos como PT, FT, etc.
            "instrumento_general",  # Captura genéricamente [A-Z]{1,4}-...
            
            # --- Patrones Eléctricos Comunes ---
            "componente_electrico_comun", # BUS, LC, UPS
            "uninterruptible_power_supply",
            "battery_bank",
            "annunciator",
            "breaker",
            "cathodic_protection",
            "cable_tray",
            "diesel_generator",
            "emergency_generator",
            "junction_box",
            "lighting_fixture",
            "lighting_panel",
            "motor",
            "receptacle",
            "switch",
            "bus_duct",

            # --- Patrones HVAC ---
            "air_handling_unit",
            "air_conditioning",
            "exhaust_fan",
            "chiller",
            "damper",

            # --- Patrones Mecánicos Generales ---
            "compressor_blower", # Tipo 'K'
            "tank_vessel",       # Tipo 'TK'

            # --- Patrones de Instrumentos Genéricos por Letra (menor prioridad) ---
            "analyzer", "burner", "control_valve", "damper_louver", "sensor", 
            "flow", "gauge", "hand", "current", "power", "time", "level", 
            "moisture", "orifice", "pressure", "quantity", "radiation", "speed", 
            "temperature", "multivariable", "vibration", "well",

            # --- Patrones de Edificios/Estructuras ---
            "building", "room", "area", "facility", "floor", "yard"
        ]
        # Añadir el resto de patrones que no están en la lista de prioridad
        pattern_order.extend([p for p in self.patterns if p not in pattern_order])

        for name in pattern_order:
            if name not in self.patterns:
                continue
            pattern_info = self.patterns[name]
            if categories and pattern_info['category'] not in categories:
                continue
            
            matches = self.test_pattern(name, text)
            all_matches.extend(matches)
        
        # Eliminar duplicados o sub-matches (el más largo y específico gana)
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
    
    # def validate_custom_patterns(self, patterns_json: str) -> Dict[str, Any]:
        # """
        # Validate custom patterns from JSON
        
        # Args:
            # patterns_json: JSON string with patterns
            
        # Returns:
            # Validation results
        # """
        # result = {
            # 'valid': True,
            # 'patterns': {},
            # 'error': None
        # }
        
        # try:
            # patterns = json.loads(patterns_json)
            
            # if not isinstance(patterns, dict):
                # result['valid'] = False
                # result['error'] = "Patterns must be a JSON object"
                # return result
            
            # for name, regex in patterns.items():
                # validation = self.validate_pattern(regex)
                # result['patterns'][name] = validation
                
                # if not validation['valid']:
                    # result['valid'] = False
                    
        # except json.JSONDecodeError as e:
            # result['valid'] = False
            # result['error'] = f"Invalid JSON: {e}"
            
        # return result
        
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
