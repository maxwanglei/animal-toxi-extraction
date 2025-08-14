import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PMCXMLParser:
    """Parse PMC XML files and extract relevant sections"""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        # Detect default namespace, if any (e.g., '{http://jats.nlm.nih.gov}article')
        self.ns: Optional[Dict[str, str]] = None
        if self.root.tag.startswith("{"):
            uri = self.root.tag[1:].split("}")[0]
            self.ns = {"n": uri}
        logger.debug("PMCXML root tag: %s ns=%s", self.root.tag, self.ns)
        self.pmid = self._extract_pmid()

    def _extract_pmid(self) -> str:
        """Extract PMID from the XML"""
        # Try namespace-aware first
        pmid_elem = self._find(self.root, ".//n:article-id[@pub-id-type='pmid']")
        if pmid_elem is None or not (pmid_elem.text or '').strip():
            pmid_elem = self._find(self.root, ".//n:pub-id[@pub-id-type='pmid']")
        # Fallback: search by localname across any namespaces
        if pmid_elem is None or not (pmid_elem.text or '').strip():
            for el in self._findall_by_localname(self.root, "article-id") + self._findall_by_localname(self.root, "pub-id"):
                if el.get("pub-id-type") == "pmid" and (el.text or '').strip():
                    return el.text
            return "unknown"
        return pmid_elem.text

    def extract_tables(self) -> List[Dict[str, Any]]:
        """Extract all tables from the XML"""
        tables: List[Dict[str, Any]] = []
        table_wraps = self._findall(self.root, ".//n:table-wrap")
        if not table_wraps:
            table_wraps = self._findall_by_localname(self.root, "table-wrap")
        for table_wrap in table_wraps:
            table_data: Dict[str, Any] = {
                'id': table_wrap.get('id', ''),
                'label': '',
                'caption': '',
                'content': '',
                'footer': ''
            }
            label_elem = self._find(table_wrap, ".//n:label") or self._find_first_by_localname(table_wrap, "label")
            if label_elem is not None and label_elem.text:
                table_data['label'] = label_elem.text
            caption_elem = self._find(table_wrap, ".//n:caption") or self._find_first_by_localname(table_wrap, "caption")
            if caption_elem is not None:
                table_data['caption'] = self._extract_text(caption_elem)
            table_elem = self._find(table_wrap, ".//n:table") or self._find_first_by_localname(table_wrap, "table")
            if table_elem is not None:
                table_data['content'] = self._table_to_text(table_elem)
            footer_elem = self._find(table_wrap, ".//n:table-wrap-foot") or self._find_first_by_localname(table_wrap, "table-wrap-foot")
            if footer_elem is not None:
                table_data['footer'] = self._extract_text(footer_elem)
            tables.append(table_data)
        return tables

    def extract_body_sections(self) -> List[Dict[str, str]]:
        """Extract body text sections"""
        sections: List[Dict[str, str]] = []
        body = self._find(self.root, ".//n:body") or self._find_first_by_localname(self.root, "body")
        if body is None:
            logger.debug("PMCXML: no body element found")
            return sections
        secs = self._findall(body, ".//n:sec")
        if not secs:
            secs = self._findall_by_localname(body, "sec")
        for sec in secs:
            section_data = {'title': '', 'content': ''}
            title_elem = self._find(sec, ".//n:title") or self._find_first_by_localname(sec, "title")
            if title_elem is not None and title_elem.text:
                section_data['title'] = title_elem.text
            section_data['content'] = self._extract_text(sec)
            sections.append(section_data)
        return sections

    def extract_full_body_text(self) -> str:
        """Extract full concatenated text from body element (namespace-agnostic)."""
        body = self._find(self.root, ".//n:body") or self._find_first_by_localname(self.root, "body")
        if body is None:
            logger.debug("PMCXML: no body found for full text")
            return ""
        text = self._extract_text(body)
        return text

    def extract_all_paragraph_text(self) -> str:
        """Fallback: concatenate all paragraph (<p>) texts found anywhere in the document."""
        ps = self._findall_by_localname(self.root, "p")
        logger.debug("PMCXML: paragraph elements found=%d", len(ps))
        chunks: List[str] = []
        for p in ps:
            t = self._extract_text(p)
            if t:
                chunks.append(t)
        return "\n\n".join(chunks)

    def extract_document_text(self) -> str:
        """Ultimate fallback: extract all text from the entire XML document."""
        text = self._extract_text(self.root)
        logger.debug("PMCXML: document text length=%d", len(text))
        return text

    def debug_tag_sample(self, limit: int = 20) -> None:
        """Log a small sample of tag localnames to help diagnose unexpected structures."""
        names: List[str] = []
        for i, e in enumerate(self.root.iter()):
            if i >= limit:
                break
            names.append(self._localname(e.tag))
        logger.debug("PMCXML: first %d tags: %s", len(names), ", ".join(names))

    def _table_to_text(self, table_elem) -> str:
        """Convert table element to readable text"""
        rows: List[str] = []
        theads = self._findall(table_elem, ".//n:thead") or self._findall_by_localname(table_elem, "thead")
        for thead in theads:
            trs = self._findall(thead, ".//n:tr") or self._findall_by_localname(thead, "tr")
            for tr in trs:
                row = []
                ths = self._findall(tr, ".//n:th") or self._findall_by_localname(tr, "th")
                tds = self._findall(tr, ".//n:td") or self._findall_by_localname(tr, "td")
                for cell in ths + tds:
                    row.append(self._extract_text(cell))
                rows.append(" | ".join(row))
        tbodys = self._findall(table_elem, ".//n:tbody") or self._findall_by_localname(table_elem, "tbody")
        for tbody in tbodys:
            trs = self._findall(tbody, ".//n:tr") or self._findall_by_localname(tbody, "tr")
            for tr in trs:
                row = []
                tds = self._findall(tr, ".//n:td") or self._findall_by_localname(tr, "td")
                ths = self._findall(tr, ".//n:th") or self._findall_by_localname(tr, "th")
                for cell in tds + ths:
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

    # --- Namespace-aware helpers ---
    def _find(self, elem, path: str):
        if self.ns:
            return elem.find(path, self.ns)
        # Fallback: try without namespace prefix
        return elem.find(path.replace("n:", ""))

    def _findall(self, elem, path: str):
        if self.ns:
            return elem.findall(path, self.ns)
        return elem.findall(path.replace("n:", ""))

    # --- Localname-based helpers (namespace-agnostic) ---
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
