"""
BioC format parser for PMC XML files.
Handles the collection/document/passage structure used by some PMC exports.
"""

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple
import html
import logging
import re

logger = logging.getLogger(__name__)


class BioCParser:
    """Parse BioC format XML files and extract relevant sections"""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        logger.debug("BioC root tag: %s", self.root.tag)
        
        # Find the document element
        self.document = self.root.find('.//document')
        if self.document is None:
            raise ValueError("No document element found in BioC XML")
            
        self.pmid = self._extract_pmid()

    def _extract_pmid(self) -> str:
        """Extract PMID from BioC XML"""
        # First try document ID
        doc_id = self.document.find('id')
        if doc_id is not None and doc_id.text:
            pmid_text = doc_id.text.strip()
            if pmid_text.startswith('PMC'):
                # Try to find actual PMID in infon elements
                for passage in self.document.findall('.//passage'):
                    for infon in passage.findall('infon'):
                        if infon.get('key') == 'article-id_pmid':
                            return infon.text.strip() if infon.text else pmid_text
                return pmid_text
            return pmid_text
        return "unknown"

    def extract_tables(self) -> List[Dict[str, Any]]:
        """Extract all tables from BioC XML"""
        tables: List[Dict[str, Any]] = []
        
        # Find all passages with table content
        for passage in self.document.findall('.//passage'):
            # Check if this passage contains table data
            section_type = None
            passage_type = None
            file_info = None
            table_id = None
            
            for infon in passage.findall('infon'):
                key = infon.get('key', '')
                if key == 'section_type':
                    section_type = infon.text
                elif key == 'type':
                    passage_type = infon.text
                elif key == 'file':
                    file_info = infon.text
                elif key == 'id':
                    table_id = infon.text
                    
            # Process table passages
            if section_type == 'TABLE':
                table_data: Dict[str, Any] = {
                    'id': table_id or file_info or '',
                    'label': '',
                    'caption': '',
                    'content': '',
                    'footer': ''
                }
                
                text_elem = passage.find('text')
                text_content = text_elem.text if text_elem is not None and text_elem.text else ''
                
                if passage_type == 'table_caption':
                    table_data['caption'] = text_content
                elif passage_type == 'table':
                    # Check for embedded XML table
                    xml_infon = None
                    for infon in passage.findall('infon'):
                        if infon.get('key') == 'xml':
                            xml_infon = infon
                            break
                    
                    if xml_infon is not None and xml_infon.text:
                        # Parse embedded table XML
                        try:
                            # Decode HTML entities
                            xml_content = html.unescape(xml_infon.text)
                            table_element = ET.fromstring(xml_content)
                            table_data['content'] = self._table_to_text(table_element)
                            table_data['raw_xml'] = xml_content
                        except ET.ParseError as e:
                            logger.warning(f"Failed to parse embedded table XML: {e}")
                            table_data['content'] = text_content
                    else:
                        table_data['content'] = text_content
                elif passage_type == 'table_foot':
                    table_data['footer'] = text_content
                
                # Group table parts by ID
                existing_table = None
                for existing in tables:
                    if existing['id'] == table_data['id']:
                        existing_table = existing
                        break
                
                if existing_table:
                    # Merge with existing table
                    if table_data['caption']:
                        existing_table['caption'] = table_data['caption']
                    if table_data['content']:
                        existing_table['content'] = table_data['content']
                        if 'raw_xml' in table_data:
                            existing_table['raw_xml'] = table_data['raw_xml']
                    if table_data['footer']:
                        existing_table['footer'] = table_data['footer']
                else:
                    tables.append(table_data)
        
        logger.debug(f"BioC parser found {len(tables)} tables")
        return tables

    def _table_to_text(self, table_element: ET.Element) -> str:
        """Convert table XML element to readable text"""
        lines = []
        
        # Process header
        thead = table_element.find('.//thead')
        if thead is not None:
            for row in thead.findall('.//tr'):
                cells = []
                for cell in row.findall('.//th'):
                    cell_text = ''.join(cell.itertext()).strip()
                    cells.append(cell_text)
                if cells:
                    lines.append('\t'.join(cells))
        
        # Process body
        tbody = table_element.find('.//tbody')
        if tbody is not None:
            for row in tbody.findall('.//tr'):
                cells = []
                for cell in row.findall('.//td'):
                    cell_text = ''.join(cell.itertext()).strip()
                    cells.append(cell_text)
                if cells:
                    lines.append('\t'.join(cells))
        
        return '\n'.join(lines)

    def extract_body_sections(self) -> List[str]:
        """Extract body sections from BioC XML"""
        sections = []
        current_section = []
        
        for passage in self.document.findall('.//passage'):
            section_type = None
            passage_type = None
            
            for infon in passage.findall('infon'):
                key = infon.get('key', '')
                if key == 'section_type':
                    section_type = infon.text
                elif key == 'type':
                    passage_type = infon.text
            
            # Skip abstract and title sections, focus on methods/results
            if section_type in ['METHODS', 'RESULTS', 'INTRO', 'DISCUSS']:
                text_elem = passage.find('text')
                if text_elem is not None and text_elem.text:
                    text_content = text_elem.text.strip()
                    
                    if passage_type and 'title' in passage_type:
                        # Section header
                        if current_section:
                            sections.append('\n'.join(current_section))
                            current_section = []
                        current_section.append(f"## {text_content}")
                    elif passage_type == 'paragraph':
                        current_section.append(text_content)
        
        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))
        
        logger.debug(f"BioC parser found {len(sections)} body sections")
        return sections
    
    def extract_full_body_text(self) -> str:
        """Extract all body text content as a single string"""
        sections = self.extract_body_sections()
        return '\n\n'.join(sections)

    def extract_all_paragraph_text(self) -> List[str]:
        """Extract all paragraph text for compatibility with enhanced pipeline"""
        try:
            if not self.root:
                return []
            
            paragraphs = []
            
            # Extract from all passages as paragraphs
            for document in self.root.findall('.//document'):
                for passage in document.findall('.//passage'):
                    passage_text = passage.findtext('text', '').strip()
                    if passage_text:
                        # Split into logical paragraphs if text is very long
                        if len(passage_text) > 500:
                            # Split on double newlines or sentence boundaries
                            parts = re.split(r'\n\n+|\. [A-Z]', passage_text)
                            for part in parts:
                                part = part.strip()
                                if len(part) > 50:  # Only include substantial paragraphs
                                    paragraphs.append(part)
                        else:
                            paragraphs.append(passage_text)
            
            return paragraphs
            
        except Exception as e:
            logger.error(f"Failed to extract paragraph text: {e}")
            return []

    def parse(self) -> Tuple[List[Dict[str, Any]], List[str], str]:
        """Parse the BioC XML and return tables, body sections, and PMID"""
        tables = self.extract_tables()
        body_sections = self.extract_body_sections()
        return tables, body_sections, self.pmid