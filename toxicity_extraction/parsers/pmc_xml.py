import xml.etree.ElementTree as ET
from typing import Any, Dict, List


class PMCXMLParser:
    """Parse PMC XML files and extract relevant sections"""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.pmid = self._extract_pmid()

    def _extract_pmid(self) -> str:
        """Extract PMID from the XML"""
        pmid_elem = self.root.find(".//article-id[@pub-id-type='pmid']")
        if pmid_elem is not None and pmid_elem.text:
            return pmid_elem.text
        pmid_elem = self.root.find(".//pub-id[@pub-id-type='pmid']")
        if pmid_elem is not None and pmid_elem.text:
            return pmid_elem.text
        return "unknown"

    def extract_tables(self) -> List[Dict[str, Any]]:
        """Extract all tables from the XML"""
        tables: List[Dict[str, Any]] = []
        for table_wrap in self.root.findall(".//table-wrap"):
            table_data: Dict[str, Any] = {
                'id': table_wrap.get('id', ''),
                'label': '',
                'caption': '',
                'content': '',
                'footer': ''
            }
            label_elem = table_wrap.find(".//label")
            if label_elem is not None and label_elem.text:
                table_data['label'] = label_elem.text
            caption_elem = table_wrap.find(".//caption")
            if caption_elem is not None:
                table_data['caption'] = self._extract_text(caption_elem)
            table_elem = table_wrap.find(".//table")
            if table_elem is not None:
                table_data['content'] = self._table_to_text(table_elem)
            footer_elem = table_wrap.find(".//table-wrap-foot")
            if footer_elem is not None:
                table_data['footer'] = self._extract_text(footer_elem)
            tables.append(table_data)
        return tables

    def extract_body_sections(self) -> List[Dict[str, str]]:
        """Extract body text sections"""
        sections: List[Dict[str, str]] = []
        body = self.root.find(".//body")
        if body is None:
            return sections
        for sec in body.findall(".//sec"):
            section_data = {'title': '', 'content': ''}
            title_elem = sec.find(".//title")
            if title_elem is not None and title_elem.text:
                section_data['title'] = title_elem.text
            section_data['content'] = self._extract_text(sec)
            sections.append(section_data)
        return sections

    def _table_to_text(self, table_elem) -> str:
        """Convert table element to readable text"""
        rows: List[str] = []
        for thead in table_elem.findall(".//thead"):
            for tr in thead.findall(".//tr"):
                row = []
                for cell in tr.findall(".//th") + tr.findall(".//td"):
                    row.append(self._extract_text(cell))
                rows.append(" | ".join(row))
        for tbody in table_elem.findall(".//tbody"):
            for tr in tbody.findall(".//tr"):
                row = []
                for cell in tr.findall(".//td") + tr.findall(".//th"):
                    row.append(self._extract_text(cell))
                rows.append(" | ".join(row))
        return "\n".join(rows)

    def _extract_text(self, element) -> str:
        """Extract all text from an element"""
        text: List[str] = []

        def extract_recursive(elem):
            if elem.text:
                text.append(elem.text)
            for child in elem:
                extract_recursive(child)
                if child.tail:
                    text.append(child.tail)

        extract_recursive(element)
        return " ".join(text).strip()
