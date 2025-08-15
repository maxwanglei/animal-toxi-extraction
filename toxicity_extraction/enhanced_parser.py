"""
Enhanced PMC XML parser with better table detection and processing.
Inspired by the HtmlTableExtractor patterns from the reference repository.
"""

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class EnhancedPMCXMLParser:
    """Enhanced PMC XML parser with better table detection and processing"""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        # Detect default namespace
        self.ns: Optional[Dict[str, str]] = None
        if self.root.tag.startswith("{"):
            uri = self.root.tag[1:].split("}")[0]
            self.ns = {"n": uri}
        logger.debug("Enhanced PMCXML root tag: %s ns=%s", self.root.tag, self.ns)
        self.pmid = self._extract_pmid()

    def _extract_pmid(self) -> str:
        """Enhanced PMID extraction with multiple fallback strategies"""
        # Try multiple strategies for PMID extraction
        strategies = [
            (".//n:article-id[@pub-id-type='pmid']", "article-id"),
            (".//n:pub-id[@pub-id-type='pmid']", "pub-id"),
            (".//n:article-meta//n:pmid", "pmid"),
            (".//n:front//n:pmid", "pmid")
        ]
        
        for xpath, localname in strategies:
            pmid_elem = self._find(self.root, xpath)
            if pmid_elem is not None and (pmid_elem.text or '').strip():
                return pmid_elem.text.strip()
                
        # Fallback: search by localname across any namespaces
        for localname in ["article-id", "pub-id", "pmid"]:
            for el in self._findall_by_localname(self.root, localname):
                if (el.get("pub-id-type") == "pmid" or localname == "pmid") and (el.text or '').strip():
                    return el.text.strip()
                    
        # Extract from filename as last resort
        import os
        filename = os.path.basename(self.xml_path)
        pmid_match = re.search(r'PMC(\d+)', filename, re.IGNORECASE)
        if pmid_match:
            return f"PMC{pmid_match.group(1)}"
            
        return "unknown"

    def extract_tables_enhanced(self) -> List[Dict[str, Any]]:
        """Enhanced table extraction with better content processing"""
        tables: List[Dict[str, Any]] = []
        
        # Strategy 1: Standard table-wrap elements
        table_wraps = self._findall(self.root, ".//n:table-wrap")
        if not table_wraps:
            table_wraps = self._findall_by_localname(self.root, "table-wrap")
            
        for table_wrap in table_wraps:
            table_data = self._process_table_wrap(table_wrap)
            if table_data:
                tables.append(table_data)
                
        # Strategy 2: Standalone table elements
        standalone_tables = self._findall(self.root, ".//n:table")
        if not standalone_tables:
            standalone_tables = self._findall_by_localname(self.root, "table")
            
        for i, table_elem in enumerate(standalone_tables):
            # Skip if already processed as part of table-wrap
            if any(table_elem in self._findall(tw, ".//n:table") or 
                   table_elem in self._findall_by_localname(tw, "table") 
                   for tw in table_wraps):
                continue
                
            table_data = self._process_standalone_table(table_elem, i)
            if table_data:
                tables.append(table_data)
                
        # Strategy 3: Array elements (common in some PMC formats)
        array_elements = self._findall_by_localname(self.root, "array")
        for i, array_elem in enumerate(array_elements):
            table_data = self._process_array_element(array_elem, i)
            if table_data:
                tables.append(table_data)
                
        logger.debug("Enhanced table extraction found %d tables", len(tables))
        return tables

    def _process_table_wrap(self, table_wrap) -> Optional[Dict[str, Any]]:
        """Process a table-wrap element"""
        try:
            table_data: Dict[str, Any] = {
                'id': table_wrap.get('id', f'table_wrap_{id(table_wrap)}'),
                'label': '',
                'caption': '',
                'content': '',
                'footer': '',
                'source_type': 'table-wrap'
            }
            
            # Extract label
            label_elem = self._find(table_wrap, ".//n:label") or self._find_first_by_localname(table_wrap, "label")
            if label_elem is not None and label_elem.text:
                table_data['label'] = label_elem.text.strip()
                
            # Extract caption with enhanced processing
            caption_elem = self._find(table_wrap, ".//n:caption") or self._find_first_by_localname(table_wrap, "caption")
            if caption_elem is not None:
                table_data['caption'] = self._extract_text_enhanced(caption_elem)
                
            # Extract table content
            table_elem = self._find(table_wrap, ".//n:table") or self._find_first_by_localname(table_wrap, "table")
            if table_elem is not None:
                table_data['content'] = self._table_to_text_enhanced(table_elem)
            else:
                # Try to find alternative table representations
                alt_content = self._extract_alternative_table_content(table_wrap)
                if alt_content:
                    table_data['content'] = alt_content
                    
            # Extract footer/notes
            footer_elems = [
                self._find(table_wrap, ".//n:table-wrap-foot"),
                self._find_first_by_localname(table_wrap, "table-wrap-foot"),
                self._find(table_wrap, ".//n:fn-group"),
                self._find_first_by_localname(table_wrap, "fn-group")
            ]
            
            for footer_elem in footer_elems:
                if footer_elem is not None:
                    footer_text = self._extract_text_enhanced(footer_elem)
                    if footer_text:
                        table_data['footer'] = footer_text
                        break
                        
            # Only return tables with meaningful content
            if table_data['content'] or table_data['caption']:
                return table_data
                
        except Exception as e:
            logger.debug("Failed to process table-wrap: %s", e)
            
        return None

    def _process_standalone_table(self, table_elem, index: int) -> Optional[Dict[str, Any]]:
        """Process a standalone table element"""
        try:
            content = self._table_to_text_enhanced(table_elem)
            if content:
                return {
                    'id': table_elem.get('id', f'standalone_table_{index}'),
                    'label': f'Table {index + 1}',
                    'caption': '',
                    'content': content,
                    'footer': '',
                    'source_type': 'standalone-table'
                }
        except Exception as e:
            logger.debug("Failed to process standalone table: %s", e)
            
        return None

    def _process_array_element(self, array_elem, index: int) -> Optional[Dict[str, Any]]:
        """Process an array element that might contain tabular data"""
        try:
            content = self._extract_text_enhanced(array_elem)
            # Check if content looks tabular
            if self._looks_like_table_content(content):
                return {
                    'id': array_elem.get('id', f'array_table_{index}'),
                    'label': f'Array {index + 1}',
                    'caption': '',
                    'content': content,
                    'footer': '',
                    'source_type': 'array'
                }
        except Exception as e:
            logger.debug("Failed to process array element: %s", e)
            
        return None

    def _extract_alternative_table_content(self, container) -> str:
        """Extract table content from alternative representations"""
        alternatives = [
            ".//n:alternatives",
            ".//n:graphic",
            ".//n:inline-graphic",
            ".//n:textual-form"
        ]
        
        for alt_xpath in alternatives:
            alt_elem = self._find(container, alt_xpath)
            if alt_elem is not None:
                content = self._extract_text_enhanced(alt_elem)
                if content and self._looks_like_table_content(content):
                    return content
                    
        return ""

    def _looks_like_table_content(self, content: str) -> bool:
        """Heuristic to determine if content looks like tabular data"""
        if not content or len(content.strip()) < 20:
            return False
            
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False
            
        # Check for table indicators
        table_indicators = [
            # Has multiple columns separated by tabs, pipes, or multiple spaces
            any('\t' in line or '|' in line or re.search(r'\s{3,}', line) for line in lines[:5]),
            # Contains numerical data patterns
            bool(re.search(r'\d+\.?\d*\s*[±+\-]\s*\d+\.?\d*', content)),
            # Contains measurement units
            bool(re.search(r'\d+\.?\d*\s*(mg|g|kg|ml|l|%|u/l|mg/dl|μg)', content, re.IGNORECASE)),
            # Contains table-like headers
            any(keyword in content.lower() for keyword in ['dose', 'group', 'treatment', 'control', 'mean', 'std'])
        ]
        
        return sum(table_indicators) >= 2

    def _table_to_text_enhanced(self, table_elem) -> str:
        """Enhanced table to text conversion with better formatting"""
        try:
            rows: List[str] = []
            
            # Process table header
            theads = self._findall(table_elem, ".//n:thead") or self._findall_by_localname(table_elem, "thead")
            for thead in theads:
                thead_rows = self._process_table_rows(thead, is_header=True)
                rows.extend(thead_rows)
                
            # Process table body
            tbodys = self._findall(table_elem, ".//n:tbody") or self._findall_by_localname(table_elem, "tbody")
            for tbody in tbodys:
                tbody_rows = self._process_table_rows(tbody, is_header=False)
                rows.extend(tbody_rows)
                
            # If no thead/tbody, process direct tr elements
            if not theads and not tbodys:
                direct_rows = self._process_table_rows(table_elem, is_header=False)
                rows.extend(direct_rows)
                
            # Format as markdown table if it looks structured
            if rows and self._is_structured_table(rows):
                return self._format_as_markdown_table(rows)
            else:
                return '\n'.join(rows)
                
        except Exception as e:
            logger.debug("Enhanced table conversion failed: %s", e)
            return self._extract_text_enhanced(table_elem)

    def _process_table_rows(self, container, is_header: bool = False) -> List[str]:
        """Process table rows with enhanced cell handling"""
        rows = []
        trs = self._findall(container, ".//n:tr") or self._findall_by_localname(container, "tr")
        
        for tr in trs:
            row_cells = []
            
            # Get all cell elements (th and td)
            cells = (self._findall(tr, ".//n:th") or self._findall_by_localname(tr, "th")) + \
                   (self._findall(tr, ".//n:td") or self._findall_by_localname(tr, "td"))
                   
            for cell in cells:
                cell_text = self._extract_text_enhanced(cell)
                
                # Handle spanning cells
                colspan = int(cell.get('colspan', '1'))
                rowspan = int(cell.get('rowspan', '1'))
                
                if colspan > 1 or rowspan > 1:
                    cell_text += f" [span: {colspan}x{rowspan}]"
                    
                row_cells.append(cell_text)
                
            if row_cells:
                rows.append(' | '.join(row_cells))
                
        return rows

    def _is_structured_table(self, rows: List[str]) -> bool:
        """Check if rows represent a structured table"""
        if len(rows) < 2:
            return False
            
        # Check if rows have consistent column counts
        col_counts = [len(row.split(' | ')) for row in rows]
        return len(set(col_counts)) <= 2  # Allow some variation

    def _format_as_markdown_table(self, rows: List[str]) -> str:
        """Format rows as a proper markdown table"""
        if not rows:
            return ""
            
        md_rows = []
        header_added = False
        
        for i, row in enumerate(rows):
            md_rows.append('| ' + row + ' |')
            
            # Add markdown separator after first row
            if not header_added:
                col_count = len(row.split(' | '))
                md_rows.append('| ' + ' | '.join(['---'] * col_count) + ' |')
                header_added = True
                
        return '\n'.join(md_rows)

    def _extract_text_enhanced(self, element) -> str:
        """Enhanced text extraction with better handling of nested elements"""
        if element is None:
            return ""
            
        text_parts: List[str] = []

        def extract_recursive(elem):
            # Add element text
            if elem.text:
                text_parts.append(elem.text.strip())
                
            for child in elem:
                # Handle special elements
                if self._localname(child.tag) in ['italic', 'bold', 'sup', 'sub']:
                    child_text = self._extract_text_enhanced(child)
                    if child_text:
                        text_parts.append(child_text)
                elif self._localname(child.tag) == 'break':
                    text_parts.append('\n')
                else:
                    extract_recursive(child)
                    
                # Add tail text
                if child.tail:
                    text_parts.append(child.tail.strip())

        extract_recursive(element)
        
        # Clean up and join
        result = ' '.join(text_parts)
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        return result.strip()

    def extract_body_sections_enhanced(self) -> List[Dict[str, str]]:
        """Enhanced body section extraction with better section detection"""
        sections: List[Dict[str, str]] = []
        
        body = self._find(self.root, ".//n:body") or self._find_first_by_localname(self.root, "body")
        if body is None:
            logger.debug("Enhanced PMCXML: no body element found")
            return sections
            
        # Get all section elements
        secs = self._findall(body, ".//n:sec") or self._findall_by_localname(body, "sec")
        
        for sec in secs:
            section_data = self._process_body_section(sec)
            if section_data:
                sections.append(section_data)
                
        # If no formal sections found, try to extract by headings
        if not sections:
            sections = self._extract_sections_by_headings(body)
            
        logger.debug("Enhanced body section extraction found %d sections", len(sections))
        return sections

    def _process_body_section(self, sec) -> Optional[Dict[str, str]]:
        """Process a single body section"""
        try:
            section_data = {'title': '', 'content': '', 'section_type': ''}
            
            # Extract title
            title_elem = self._find(sec, ".//n:title") or self._find_first_by_localname(sec, "title")
            if title_elem is not None and title_elem.text:
                section_data['title'] = title_elem.text.strip()
                
            # Determine section type based on title
            title_lower = section_data['title'].lower()
            if any(kw in title_lower for kw in ['result', 'finding', 'outcome']):
                section_data['section_type'] = 'results'
            elif any(kw in title_lower for kw in ['method', 'material', 'procedure']):
                section_data['section_type'] = 'methods'
            elif any(kw in title_lower for kw in ['discussion', 'conclusion']):
                section_data['section_type'] = 'discussion'
            elif any(kw in title_lower for kw in ['introduction', 'background']):
                section_data['section_type'] = 'introduction'
            else:
                section_data['section_type'] = 'other'
                
            # Extract content
            section_data['content'] = self._extract_text_enhanced(sec)
            
            # Only return sections with meaningful content
            if section_data['content'] and len(section_data['content'].strip()) > 50:
                return section_data
                
        except Exception as e:
            logger.debug("Failed to process body section: %s", e)
            
        return None

    def _extract_sections_by_headings(self, body) -> List[Dict[str, str]]:
        """Fallback: extract sections by heading elements"""
        sections = []
        
        # Find heading elements
        heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        current_section = None
        
        for elem in body.iter():
            localname = self._localname(elem.tag)
            
            if localname in heading_tags:
                # Save previous section
                if current_section and current_section['content'].strip():
                    sections.append(current_section)
                    
                # Start new section
                current_section = {
                    'title': self._extract_text_enhanced(elem),
                    'content': '',
                    'section_type': 'other'
                }
            elif current_section and localname == 'p':
                # Add paragraph to current section
                para_text = self._extract_text_enhanced(elem)
                if para_text:
                    current_section['content'] += para_text + '\n\n'
                    
        # Add final section
        if current_section and current_section['content'].strip():
            sections.append(current_section)
            
        return sections

    # Keep the existing namespace helper methods
    def _find(self, elem, path: str):
        if self.ns:
            return elem.find(path, self.ns)
        return elem.find(path.replace("n:", ""))

    def _findall(self, elem, path: str):
        if self.ns:
            return elem.findall(path, self.ns)
        return elem.findall(path.replace("n:", ""))

    @staticmethod
    def _localname(tag: str) -> str:
        if tag.startswith("{"):
            return tag.split("}", 1)[1]
        return tag

    def _find_first_by_localname(self, elem, localname: str):
        for e in elem.iter():
            if self._localname(e.tag) == localname:
                return e
        return None

    def _findall_by_localname(self, elem, localname: str):
        return [e for e in elem.iter() if self._localname(e.tag) == localname]

    # Keep compatibility with existing methods
    def extract_tables(self) -> List[Dict[str, Any]]:
        """Maintain compatibility with existing interface"""
        return self.extract_tables_enhanced()
        
    def extract_body_sections(self) -> List[Dict[str, str]]:
        """Maintain compatibility with existing interface"""
        return self.extract_body_sections_enhanced()
        
    def extract_full_body_text(self) -> str:
        """Extract full concatenated text from body element"""
        body = self._find(self.root, ".//n:body") or self._find_first_by_localname(self.root, "body")
        if body is None:
            return ""
        return self._extract_text_enhanced(body)

    def extract_all_paragraph_text(self) -> str:
        """Fallback: concatenate all paragraph texts"""
        ps = self._findall_by_localname(self.root, "p")
        chunks = []
        for p in ps:
            text = self._extract_text_enhanced(p)
            if text:
                chunks.append(text)
        return "\n\n".join(chunks)

    def extract_document_text(self) -> str:
        """Ultimate fallback: extract all text from document"""
        return self._extract_text_enhanced(self.root)