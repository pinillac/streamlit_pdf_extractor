"""
Configuration UI Component
Handles user interface for configuration management
"""

import streamlit as st
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

class ConfigurationUI:
    """Manages configuration interface and validation"""
    
    def __init__(self):
        self.default_patterns = self.load_default_patterns()
        self.pattern_templates = self.load_pattern_templates()
        
    def load_default_patterns(self) -> Dict[str, str]:
        """Load default regex patterns"""
        return {
            'bomba': r'\b\d{3}P\d{4}(?:[A-Z](?:/[A-Z])?)?\b',
            'horno': r'\b\d{3}F\d{4}(?:[A-Z](?:/[A-Z])?)?\b',
            'intercambiador': r'\b\d{3}E\d{4}[A-Z]?\b',
            'recipiente': r'\b\d{3}V\d{4}[A-Z]?\b',
            'compresor': r'\b\d{3}C\d{4}[A-Z]?\b',
            'instrumento': r'(?<![A-Z0-9-])(PT|FT|TE|LT|LIC|PI|FIC)-\d{3,5}[A-Z]?(?![A-Z0-9-])',
           # 'instrumento': r"(?:^|\s|\b)[A-Z]{2,4}-\d{3,5}[A-Z]?(?:\b|\s|$)",
            'tuberia': r'\b\d{1,2}"-[A-Z]{2,4}-\d{4,6}-[A-Z0-9-]{1,20}\b'
        }
    
    def load_pattern_templates(self) -> Dict[str, Dict[str, str]]:
        """Load pattern templates for quick selection"""
        return {
            'ISA Instruments': {
                'pressure_transmitter': r'\bPT-\d{3,5}[A-Z]?\b',
                'flow_indicator': r'\bFI-\d{3,5}[A-Z]?\b',
                'temperature_controller': r'\bTC-\d{3,5}[A-Z]?\b',
                'level_alarm': r'\bLA[HL]+-\d{3,5}[A-Z]?\b',
                'control_valve': r'\b[A-Z]CV-\d{3,5}[A-Z]?\b'
            },
            'Equipment Tags': {
                'pump_extended': r'\b[A-Z]{2,3}-P-\d{4}[A-Z]?\b',
                'vessel_extended': r'\b[A-Z]{2,3}-V-\d{4}[A-Z]?\b',
                'heat_exchanger_extended': r'\b[A-Z]{2,3}-E-\d{4}[A-Z]?\b',
                'compressor_extended': r'\b[A-Z]{2,3}-C-\d{4}[A-Z]?\b'
            },
            'Piping Systems': {
                'pipe_line_number': r'\b\d{4}-[A-Z]{2,4}-\d{3,4}\b',
                'pipe_spec': r'\b[A-Z]{2}\d{2}[A-Z]\d{2}\b',
                'pipe_isometric': r'\bISO-\d{4}-\d{3}\b'
            }
        }
    
    def render_pattern_editor(self) -> Dict[str, str]:
        """Render pattern editing interface"""
        st.subheader("🔧 Pattern Configuration")
        
        # Pattern management tabs
        tab1, tab2, tab3 = st.tabs(["Default Patterns", "Custom Patterns", "Pattern Library"])
        
        patterns = {}
        
        with tab1:
            st.write("### Default Equipment Patterns")
            
            # Display default patterns with toggle
            use_defaults = st.checkbox("Use default patterns", value=True)
            
            if use_defaults:
                for pattern_name, pattern_regex in self.default_patterns.items():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        use_pattern = st.checkbox(pattern_name.title(), value=True, key=f"use_{pattern_name}")
                    
                    with col2:
                        if use_pattern:
                            patterns[pattern_name] = pattern_regex
                            st.code(pattern_regex, language='regex')
        
        with tab2:
            st.write("### Custom Pattern Editor")
            
            # Pattern input
            col1, col2 = st.columns([1, 2])
            
            with col1:
                pattern_name = st.text_input("Pattern Name", placeholder="e.g., custom_valve")
            
            with col2:
                pattern_regex = st.text_input("Regex Pattern", placeholder=r"\b[A-Z]{3}\d{4}\b")
            
            # Test pattern
            if pattern_name and pattern_regex:
                test_string = st.text_area("Test String", placeholder="Enter text to test pattern...")
                
                if st.button("Test Pattern"):
                    self.test_pattern(pattern_regex, test_string)
                
                if st.button("Add Pattern"):
                    patterns[pattern_name] = pattern_regex
                    st.success(f"Pattern '{pattern_name}' added!")
            
            # Display current custom patterns
            if 'custom_patterns' in st.session_state:
                st.write("#### Current Custom Patterns")
                for name, regex in st.session_state.custom_patterns.items():
                    col1, col2, col3 = st.columns([2, 4, 1])
                    
                    with col1:
                        st.text(name)
                    with col2:
                        st.code(regex, language='regex')
                    with col3:
                        if st.button("🗑️", key=f"del_{name}"):
                            del st.session_state.custom_patterns[name]
                            st.rerun()
        
        with tab3:
            st.write("### Pattern Template Library")
            
            # Display pattern templates
            for category, templates in self.pattern_templates.items():
                with st.expander(f"📁 {category}"):
                    for template_name, template_regex in templates.items():
                        col1, col2, col3 = st.columns([2, 4, 1])
                        
                        with col1:
                            st.text(template_name.replace('_', ' ').title())
                        with col2:
                            st.code(template_regex, language='regex')
                        with col3:
                            if st.button("Use", key=f"use_template_{template_name}"):
                                patterns[template_name] = template_regex
                                st.success(f"Added {template_name} pattern")
        
        return patterns
    
    def test_pattern(self, pattern: str, test_string: str):
        """Test regex pattern against string"""
        try:
            compiled_pattern = re.compile(pattern)
            matches = list(compiled_pattern.finditer(test_string))
            
            if matches:
                st.success(f"✅ Found {len(matches)} matches!")
                
                # Highlight matches
                highlighted_text = test_string
                for match in reversed(matches):  # Reverse to maintain positions
                    start, end = match.span()
                    highlighted_text = (
                        highlighted_text[:start] + 
                        f"**[{highlighted_text[start:end]}]**" + 
                        highlighted_text[end:]
                    )
                
                st.markdown(f"Highlighted: {highlighted_text}")
                
                # Show match details
                st.write("Match Details:")
                for i, match in enumerate(matches, 1):
                    st.write(f"{i}. '{match.group()}' at position {match.span()}")
            else:
                st.warning("No matches found")
                
        except re.error as e:
            st.error(f"❌ Invalid regex pattern: {e}")
    
    def render_performance_settings(self) -> Dict[str, Any]:
        """Render performance configuration interface"""
        st.subheader("⚡ Performance Settings")
        
        settings = {}
        
        # Memory settings
        with st.expander("💾 Memory Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                settings['memory_limit_mb'] = st.number_input(
                    "Memory Limit (MB)",
                    min_value=256,
                    max_value=8192,
                    value=1024,
                    step=256,
                    help="Maximum memory usage for processing"
                )
                
                settings['chunk_size'] = st.slider(
                    "Chunk Size (pages)",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="Number of pages to process at once"
                )
            
            with col2:
                settings['gc_threshold'] = st.slider(
                    "Garbage Collection Threshold (%)",
                    min_value=50,
                    max_value=95,
                    value=80,
                    help="Trigger GC when memory usage reaches this percentage"
                )
                
                settings['enable_memory_monitoring'] = st.checkbox(
                    "Enable Memory Monitoring",
                    value=True,
                    help="Monitor memory usage during processing"
                )
        
        # Processing settings
        with st.expander("🔄 Processing Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                settings['processing_timeout'] = st.number_input(
                    "Processing Timeout (seconds)",
                    min_value=60,
                    max_value=3600,
                    value=300,
                    step=60,
                    help="Maximum time for processing a single file"
                )
                
                settings['max_workers'] = st.slider(
                    "Max Worker Threads",
                    min_value=1,
                    max_value=16,
                    value=4,
                    help="Number of parallel workers for batch processing"
                )
            
            with col2:
                settings['retry_attempts'] = st.number_input(
                    "Retry Attempts",
                    min_value=0,
                    max_value=5,
                    value=2,
                    help="Number of retry attempts for failed files"
                )
                
                settings['enable_caching'] = st.checkbox(
                    "Enable Result Caching",
                    value=True,
                    help="Cache results to avoid reprocessing"
                )
        
        # Advanced settings
        with st.expander("🛠️ Advanced Settings"):
            settings['log_level'] = st.selectbox(
                "Logging Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1
            )
            
            settings['save_intermediate_results'] = st.checkbox(
                "Save Intermediate Results",
                value=False,
                help="Save results after each file (slower but safer)"
            )
            
            settings['validate_output'] = st.checkbox(
                "Validate Output",
                value=True,
                help="Perform validation checks on extracted data"
            )
        
        return settings
    
    def render_preset_selector(self) -> Optional[Dict[str, Any]]:
        """Render preset configuration selector"""
        st.subheader("📋 Configuration Presets")
        
        presets = {
            'Fast Processing': {
                'description': 'Optimized for speed with moderate accuracy',
                'config': {
                    'memory_limit_mb': 2048,
                    'chunk_size': 50,
                    'max_workers': 8,
                    'processing_timeout': 180
                }
            },
            'Memory Efficient': {
                'description': 'Minimal memory usage for large files',
                'config': {
                    'memory_limit_mb': 512,
                    'chunk_size': 5,
                    'max_workers': 2,
                    'processing_timeout': 600
                }
            },
            'High Accuracy': {
                'description': 'Maximum accuracy with detailed extraction',
                'config': {
                    'memory_limit_mb': 4096,
                    'chunk_size': 10,
                    'max_workers': 4,
                    'processing_timeout': 900,
                    'validate_output': True
                }
            },
            'Balanced': {
                'description': 'Balanced performance and accuracy',
                'config': {
                    'memory_limit_mb': 1024,
                    'chunk_size': 20,
                    'max_workers': 4,
                    'processing_timeout': 300
                }
            }
        }
        
        # Preset selection
        selected_preset = st.selectbox(
            "Select Preset",
            ["None"] + list(presets.keys())
        )
        
        if selected_preset != "None":
            preset = presets[selected_preset]
            
            # Display preset details
            st.info(f"📝 {preset['description']}")
            
            # Show configuration
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Configuration:**")
                for key, value in preset['config'].items():
                    st.write(f"- {key}: {value}")
            
            with col2:
                if st.button("Apply Preset", type="primary"):
                    return preset['config']
        
        return None
    
    def render_configuration_import_export(self):
        """Render configuration import/export interface"""
        st.subheader("💾 Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Export Configuration")
            
            export_format = st.selectbox("Export Format", ["JSON", "YAML"])
            
            if st.button("Export Current Config"):
                config = st.session_state.get('config', {})
                
                if export_format == "JSON":
                    config_str = json.dumps(config, indent=2)
                    mime_type = "application/json"
                    file_ext = "json"
                else:
                    config_str = yaml.dump(config, default_flow_style=False)
                    mime_type = "application/x-yaml"
                    file_ext = "yaml"
                
                st.download_button(
                    label=f"Download {export_format} Config",
                    data=config_str,
                    file_name=f"pdf_extractor_config.{file_ext}",
                    mime=mime_type
                )
        
        with col2:
            st.write("### Import Configuration")
            
            uploaded_config = st.file_uploader(
                "Choose configuration file",
                type=['json', 'yaml', 'yml']
            )
            
            if uploaded_config:
                try:
                    content = uploaded_config.read().decode('utf-8')
                    
                    if uploaded_config.name.endswith('.json'):
                        imported_config = json.loads(content)
                    else:
                        imported_config = yaml.safe_load(content)
                    
                    st.success("✅ Configuration loaded successfully!")
                    
                    # Preview configuration
                    with st.expander("Preview Imported Configuration"):
                        st.json(imported_config)
                    
                    if st.button("Apply Imported Configuration"):
                        st.session_state.config.update(imported_config)
                        st.success("Configuration applied!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading configuration: {str(e)}")
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration settings
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check memory settings
        if config.get('memory_limit_mb', 0) < 256:
            validation['errors'].append("Memory limit too low (minimum 256MB)")
            validation['valid'] = False
        
        if config.get('chunk_size', 0) < 1:
            validation['errors'].append("Chunk size must be at least 1")
            validation['valid'] = False
        
        # Check processing settings
        if config.get('max_workers', 1) > 16:
            validation['warnings'].append("High number of workers may cause system instability")
        
        if config.get('processing_timeout', 300) < 60:
            validation['warnings'].append("Very short timeout may cause processing failures")
        
        # Validate patterns
        if 'custom_patterns' in config:
            for name, pattern in config['custom_patterns'].items():
                try:
                    re.compile(pattern)
                except re.error as e:
                    validation['errors'].append(f"Invalid pattern '{name}': {e}")
                    validation['valid'] = False
        
        return validation
    
    def render_help_section(self):
        """Render help and documentation section"""
        st.subheader("❓ Help & Documentation")
        
        with st.expander("Pattern Syntax Guide"):
            st.markdown("""
            ### Regular Expression Quick Reference
            
            - `\\b` - Word boundary
            - `\\d` - Digit (0-9)
            - `\\w` - Word character (a-z, A-Z, 0-9, _)
            - `[A-Z]` - Character class (uppercase letters)
            - `{3}` - Exactly 3 occurrences
            - `{2,4}` - Between 2 and 4 occurrences
            - `?` - Optional (0 or 1 occurrence)
            - `+` - One or more occurrences
            - `*` - Zero or more occurrences
            - `|` - Alternation (OR)
            - `()` - Grouping
            
            ### Common Equipment Pattern Examples
            
            - **Pump**: `\\b\\d{3}P\\d{4}[A-Z]?\\b` - Matches: 123P4567, 123P4567A
            - **Instrument**: `\\b[A-Z]{2,4}-\\d{3,5}\\b` - Matches: PT-123, FIC-12345
            - **Pipe**: `\\b\\d{1,2}"-[A-Z]{2,4}-\\d{4,6}\\b` - Matches: 6"-PT-1234
            """)
        
        with st.expander("Performance Optimization Tips"):
            st.markdown("""
            ### Optimization Guidelines
            
            1. **Memory Settings**:
               - Use higher memory limits for large files
               - Reduce chunk size if running out of memory
               - Enable garbage collection for long processing
            
            2. **Processing Speed**:
               - Increase worker threads for multiple files
               - Use larger chunks for faster processing
               - Disable validation for speed improvement
            
            3. **Pattern Optimization**:
               - Avoid complex lookarounds
               - Use specific patterns over generic ones
               - Test patterns on sample data first
            """)
        
        with st.expander("Troubleshooting"):
            st.markdown("""
            ### Common Issues and Solutions
            
            **Q: Processing is very slow**
            - A: Increase chunk size and memory limit
            
            **Q: Out of memory errors**
            - A: Reduce chunk size or process fewer files at once
            
            **Q: No tags found**
            - A: Check pattern configuration and PDF content
            
            **Q: Application crashes**
            - A: Check system resources and reduce worker threads
            """)