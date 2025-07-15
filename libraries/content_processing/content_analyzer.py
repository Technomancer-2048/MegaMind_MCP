#!/usr/bin/env python3
"""
Content Analysis Library for Enhanced Multi-Embedding Entry System
Provides intelligent document structure analysis and content type detection
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content that can be identified"""
    MARKDOWN = "markdown"
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    PLAIN_TEXT = "plain_text"
    MIXED = "mixed"

class MarkdownElementType(Enum):
    """Types of markdown elements"""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    LIST_ITEM = "list_item"
    TABLE = "table"
    BLOCKQUOTE = "blockquote"
    HORIZONTAL_RULE = "horizontal_rule"
    LINK = "link"
    IMAGE = "image"
    INLINE_CODE = "inline_code"

@dataclass
class MarkdownElement:
    """Represents a single markdown element"""
    element_type: MarkdownElementType
    content: str
    raw_content: str
    line_start: int
    line_end: int
    level: Optional[int] = None  # For headings
    language: Optional[str] = None  # For code blocks
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticBoundary:
    """Represents a semantic boundary in the document"""
    boundary_type: str  # 'heading', 'section', 'topic_shift', 'code_transition'
    position: int  # Line number
    confidence: float  # 0.0 to 1.0
    context: Optional[str] = None

@dataclass
class DocumentStructure:
    """Complete document structure analysis"""
    content_type: ContentType
    elements: List[MarkdownElement]
    semantic_boundaries: List[SemanticBoundary]
    metadata: Dict[str, Any]
    statistics: Dict[str, int]
    content_hash: str

class ContentAnalyzer:
    """
    Analyzes document content to identify structure, types, and semantic boundaries
    """
    
    def __init__(self):
        # Regex patterns for markdown elements
        self.patterns = {
            'heading': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            'code_block': re.compile(r'^```(\w*)\n(.*?)```', re.MULTILINE | re.DOTALL),
            'list_item': re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', re.MULTILINE),
            'table_row': re.compile(r'^\|(.+)\|$', re.MULTILINE),
            'blockquote': re.compile(r'^>\s+(.+)$', re.MULTILINE),
            'horizontal_rule': re.compile(r'^(---+|___+|\*\*\*+)$', re.MULTILINE),
            'link': re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
            'image': re.compile(r'!\[([^\]]*)\]\(([^)]+)\)'),
            'inline_code': re.compile(r'`([^`]+)`'),
            'emphasis': re.compile(r'(\*{1,2}|_{1,2})([^*_]+)\1'),
        }
        
        # Code language detection patterns
        self.code_patterns = {
            'python': re.compile(r'(def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import)'),
            'javascript': re.compile(r'(function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|=>\s*{)'),
            'java': re.compile(r'(public\s+class|private\s+\w+|protected\s+\w+|@\w+)'),
            'sql': re.compile(r'(SELECT\s+|INSERT\s+INTO|UPDATE\s+|DELETE\s+FROM)', re.IGNORECASE),
            'yaml': re.compile(r'^(\s*\w+):\s*(.+)$', re.MULTILINE),
            'json': re.compile(r'^\s*[{[]'),
        }
        
        logger.info("ContentAnalyzer initialized")
    
    def analyze_document_structure(self, content: str) -> DocumentStructure:
        """
        Analyze document structure and identify all elements
        
        Args:
            content: Document content to analyze
            
        Returns:
            DocumentStructure with complete analysis
        """
        lines = content.split('\n')
        
        # Detect content type
        content_type = self.detect_content_type(content)
        
        # Extract markdown elements
        elements = self.extract_markdown_elements(content)
        
        # Identify semantic boundaries
        boundaries = self.identify_semantic_boundaries(content, elements)
        
        # Calculate statistics
        statistics = self._calculate_statistics(elements)
        
        # Generate content hash
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Build metadata
        metadata = {
            'total_lines': len(lines),
            'total_characters': len(content),
            'has_code_blocks': any(e.element_type == MarkdownElementType.CODE_BLOCK for e in elements),
            'has_headings': any(e.element_type == MarkdownElementType.HEADING for e in elements),
            'primary_language': self._detect_primary_language(elements),
        }
        
        return DocumentStructure(
            content_type=content_type,
            elements=elements,
            semantic_boundaries=boundaries,
            metadata=metadata,
            statistics=statistics,
            content_hash=content_hash
        )
    
    def detect_content_type(self, text: str) -> ContentType:
        """
        Detect the primary content type of the document
        
        Args:
            text: Document text to analyze
            
        Returns:
            ContentType enum value
        """
        # Check for markdown indicators
        markdown_score = 0
        if self.patterns['heading'].search(text):
            markdown_score += 2
        if self.patterns['code_block'].search(text):
            markdown_score += 2
        if self.patterns['list_item'].search(text):
            markdown_score += 1
        if self.patterns['link'].search(text):
            markdown_score += 1
        
        # Check for code indicators
        code_score = 0
        for lang, pattern in self.code_patterns.items():
            if pattern.search(text):
                code_score += 2
        
        # Check for configuration patterns
        config_score = 0
        if text.strip().startswith('{') or text.strip().startswith('['):
            config_score += 3  # JSON
        if self.code_patterns['yaml'].search(text):
            config_score += 3  # YAML
        
        # Determine primary type
        scores = {
            ContentType.MARKDOWN: markdown_score,
            ContentType.CODE: code_score,
            ContentType.CONFIGURATION: config_score,
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return ContentType.PLAIN_TEXT
        elif markdown_score > 0 and code_score > 0:
            return ContentType.MIXED
        else:
            return max(scores.items(), key=lambda x: x[1])[0]
    
    def extract_markdown_elements(self, content: str) -> List[MarkdownElement]:
        """
        Extract all markdown elements from content
        
        Args:
            content: Document content
            
        Returns:
            List of MarkdownElement objects
        """
        elements = []
        lines = content.split('\n')
        line_positions = self._calculate_line_positions(content)
        
        # Extract headings
        for match in self.patterns['heading'].finditer(content):
            level = len(match.group(1))
            heading_text = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            element = MarkdownElement(
                element_type=MarkdownElementType.HEADING,
                content=heading_text,
                raw_content=match.group(0),
                line_start=self._get_line_number(start_pos, line_positions),
                line_end=self._get_line_number(end_pos, line_positions),
                level=level,
                metadata={'heading_level': level}
            )
            elements.append(element)
        
        # Extract code blocks
        for match in self.patterns['code_block'].finditer(content):
            language = match.group(1) or self._detect_code_language(match.group(2))
            code_content = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            element = MarkdownElement(
                element_type=MarkdownElementType.CODE_BLOCK,
                content=code_content,
                raw_content=match.group(0),
                line_start=self._get_line_number(start_pos, line_positions),
                line_end=self._get_line_number(end_pos, line_positions),
                language=language,
                metadata={'language': language, 'lines_of_code': len(code_content.split('\n'))}
            )
            elements.append(element)
        
        # Extract list items
        current_list_depth = 0
        for i, line in enumerate(lines):
            list_match = self.patterns['list_item'].match(line)
            if list_match:
                indent = len(list_match.group(1))
                marker = list_match.group(2)
                content_text = list_match.group(3)
                
                element = MarkdownElement(
                    element_type=MarkdownElementType.LIST_ITEM,
                    content=content_text,
                    raw_content=line,
                    line_start=i + 1,
                    line_end=i + 1,
                    metadata={
                        'indent_level': indent // 2,
                        'list_type': 'ordered' if marker[0].isdigit() else 'unordered',
                        'marker': marker
                    }
                )
                elements.append(element)
        
        # Extract paragraphs (content between other elements)
        elements = self._extract_paragraphs(content, elements, lines)
        
        # Sort elements by line start
        elements.sort(key=lambda x: x.line_start)
        
        return elements
    
    def identify_semantic_boundaries(self, content: str, elements: List[MarkdownElement]) -> List[SemanticBoundary]:
        """
        Identify semantic boundaries in the document
        
        Args:
            content: Document content
            elements: List of markdown elements
            
        Returns:
            List of SemanticBoundary objects
        """
        boundaries = []
        
        # Headings are natural boundaries
        for element in elements:
            if element.element_type == MarkdownElementType.HEADING:
                boundary = SemanticBoundary(
                    boundary_type='heading',
                    position=element.line_start,
                    confidence=1.0,
                    context=element.content
                )
                boundaries.append(boundary)
        
        # Code blocks create boundaries
        for element in elements:
            if element.element_type == MarkdownElementType.CODE_BLOCK:
                # Boundary before code block
                boundary_before = SemanticBoundary(
                    boundary_type='code_transition',
                    position=element.line_start,
                    confidence=0.9,
                    context='entering_code_block'
                )
                boundaries.append(boundary_before)
                
                # Boundary after code block
                boundary_after = SemanticBoundary(
                    boundary_type='code_transition',
                    position=element.line_end + 1,
                    confidence=0.9,
                    context='exiting_code_block'
                )
                boundaries.append(boundary_after)
        
        # Detect topic shifts in paragraphs
        topic_boundaries = self._detect_topic_shifts(elements)
        boundaries.extend(topic_boundaries)
        
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x.position)
        
        return boundaries
    
    def _detect_code_language(self, code: str) -> Optional[str]:
        """Detect programming language from code content"""
        for lang, pattern in self.code_patterns.items():
            if pattern.search(code):
                return lang
        return None
    
    def _calculate_line_positions(self, content: str) -> List[int]:
        """Calculate character positions for each line start"""
        positions = [0]
        for i, char in enumerate(content):
            if char == '\n':
                positions.append(i + 1)
        return positions
    
    def _get_line_number(self, char_pos: int, line_positions: List[int]) -> int:
        """Get line number from character position"""
        for i, pos in enumerate(line_positions):
            if char_pos < pos:
                return i
        return len(line_positions)
    
    def _extract_paragraphs(self, content: str, existing_elements: List[MarkdownElement], lines: List[str]) -> List[MarkdownElement]:
        """Extract paragraphs from content not covered by other elements"""
        all_elements = existing_elements.copy()
        covered_lines = set()
        
        # Mark lines covered by existing elements
        for element in existing_elements:
            for line_num in range(element.line_start, element.line_end + 1):
                covered_lines.add(line_num)
        
        # Find continuous blocks of uncovered lines
        paragraph_start = None
        paragraph_lines = []
        
        for i, line in enumerate(lines, 1):
            if i not in covered_lines and line.strip():
                if paragraph_start is None:
                    paragraph_start = i
                paragraph_lines.append(line)
            else:
                if paragraph_start is not None and paragraph_lines:
                    # Create paragraph element
                    element = MarkdownElement(
                        element_type=MarkdownElementType.PARAGRAPH,
                        content='\n'.join(paragraph_lines),
                        raw_content='\n'.join(paragraph_lines),
                        line_start=paragraph_start,
                        line_end=i - 1,
                        metadata={'word_count': sum(len(line.split()) for line in paragraph_lines)}
                    )
                    all_elements.append(element)
                    
                paragraph_start = None
                paragraph_lines = []
        
        # Handle last paragraph
        if paragraph_start is not None and paragraph_lines:
            element = MarkdownElement(
                element_type=MarkdownElementType.PARAGRAPH,
                content='\n'.join(paragraph_lines),
                raw_content='\n'.join(paragraph_lines),
                line_start=paragraph_start,
                line_end=len(lines),
                metadata={'word_count': sum(len(line.split()) for line in paragraph_lines)}
            )
            all_elements.append(element)
        
        return all_elements
    
    def _detect_topic_shifts(self, elements: List[MarkdownElement]) -> List[SemanticBoundary]:
        """Detect topic shifts within paragraphs"""
        boundaries = []
        
        # Simple heuristic: long gaps between elements might indicate topic shifts
        for i in range(1, len(elements)):
            prev_element = elements[i-1]
            curr_element = elements[i]
            
            gap = curr_element.line_start - prev_element.line_end
            if gap > 3:  # More than 3 empty lines
                boundary = SemanticBoundary(
                    boundary_type='topic_shift',
                    position=prev_element.line_end + 2,
                    confidence=0.7,
                    context='large_gap_detected'
                )
                boundaries.append(boundary)
        
        return boundaries
    
    def _calculate_statistics(self, elements: List[MarkdownElement]) -> Dict[str, int]:
        """Calculate statistics about the document elements"""
        stats = {
            'total_elements': len(elements),
            'headings': 0,
            'paragraphs': 0,
            'code_blocks': 0,
            'list_items': 0,
            'total_words': 0,
            'total_code_lines': 0,
        }
        
        for element in elements:
            if element.element_type == MarkdownElementType.HEADING:
                stats['headings'] += 1
            elif element.element_type == MarkdownElementType.PARAGRAPH:
                stats['paragraphs'] += 1
                stats['total_words'] += element.metadata.get('word_count', 0)
            elif element.element_type == MarkdownElementType.CODE_BLOCK:
                stats['code_blocks'] += 1
                stats['total_code_lines'] += element.metadata.get('lines_of_code', 0)
            elif element.element_type == MarkdownElementType.LIST_ITEM:
                stats['list_items'] += 1
        
        return stats
    
    def _detect_primary_language(self, elements: List[MarkdownElement]) -> Optional[str]:
        """Detect the primary programming language in the document"""
        language_counts = {}
        
        for element in elements:
            if element.element_type == MarkdownElementType.CODE_BLOCK and element.language:
                language_counts[element.language] = language_counts.get(element.language, 0) + 1
        
        if language_counts:
            return max(language_counts.items(), key=lambda x: x[1])[0]
        return None