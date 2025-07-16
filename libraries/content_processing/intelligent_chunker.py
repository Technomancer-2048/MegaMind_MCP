#!/usr/bin/env python3
"""
Intelligent Chunking Library for Enhanced Multi-Embedding Entry System
Provides content-aware chunking that respects semantic boundaries and token limits
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib

from .content_analyzer import MarkdownElement, MarkdownElementType, SemanticBoundary, DocumentStructure
from .sentence_splitter import SentenceSplitter

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    SEMANTIC_AWARE = "semantic_aware"
    MARKDOWN_STRUCTURE = "markdown_structure"
    FIXED_SIZE = "fixed_size"
    SLIDING_WINDOW = "sliding_window"
    HYBRID = "hybrid"

class ChunkType(Enum):
    """Types of chunks based on content"""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    LIST_SECTION = "list_section"
    TABLE = "table"
    MIXED = "mixed"
    RULE = "rule"
    EXAMPLE = "example"
    DEFINITION = "definition"

@dataclass
class Chunk:
    """Represents a semantic chunk of content"""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    line_start: int
    line_end: int
    token_count: int
    elements: List[MarkdownElement]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    semantic_coherence: float = 0.0
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)

@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior"""
    max_tokens: int = 512
    min_tokens: int = 50
    overlap_tokens: int = 50
    respect_boundaries: bool = True
    preserve_code_blocks: bool = True
    preserve_headings: bool = True
    quality_threshold: float = 0.7
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_AWARE

class IntelligentChunker:
    """
    Performs intelligent content chunking that respects semantic boundaries
    and optimizes for embedding model token limits
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.sentence_splitter = SentenceSplitter()
        self.table_min_rows = 2  # Minimum rows to consider as table
        
        # Token estimation (rough approximation - 1 word ≈ 1.3 tokens)
        self.token_multiplier = 1.3
        
        # Patterns for special content detection
        self.patterns = {
            'rule': re.compile(r'^(Rule|Principle|Guideline|Policy)[\s:]+', re.IGNORECASE),
            'example': re.compile(r'^(Example|E\.g\.|For example|Such as)[\s:]+', re.IGNORECASE),
            'definition': re.compile(r'^(Definition|Define|Meaning)[\s:]+', re.IGNORECASE),
            'important': re.compile(r'^(Important|Note|Warning|Caution|Critical)[\s:]+', re.IGNORECASE),
        }
        
        logger.info(f"IntelligentChunker initialized with strategy: {self.config.strategy}")
    
    def chunk_document(self, document_structure: DocumentStructure) -> List[Chunk]:
        """
        Chunk a document based on its structure and configured strategy
        
        Args:
            document_structure: Analyzed document structure
            
        Returns:
            List of Chunk objects
        """
        if self.config.strategy == ChunkingStrategy.SEMANTIC_AWARE:
            return self.chunk_by_semantic_boundaries(
                document_structure.elements,
                document_structure.semantic_boundaries
            )
        elif self.config.strategy == ChunkingStrategy.MARKDOWN_STRUCTURE:
            return self.chunk_by_markdown_structure(document_structure.elements)
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            return self._hybrid_chunking(document_structure)
        else:
            return self._fixed_size_chunking(document_structure.elements)
    
    def chunk_by_semantic_boundaries(self, 
                                   elements: List[MarkdownElement],
                                   boundaries: List[SemanticBoundary]) -> List[Chunk]:
        """
        Chunk content respecting semantic boundaries
        
        Args:
            elements: List of markdown elements
            boundaries: List of semantic boundaries
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk_elements = []
        current_tokens = 0
        chunk_start_line = 1
        
        for element in elements:
            element_tokens = self._estimate_tokens(element.content)
            
            # Check if element should start a new chunk
            should_split = self._should_split_at_element(
                element, current_tokens, element_tokens, boundaries
            )
            
            if should_split and current_chunk_elements:
                # Create chunk from current elements
                chunk = self._create_chunk_from_elements(
                    current_chunk_elements,
                    chunk_start_line
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_elements = [element]
                current_tokens = element_tokens
                chunk_start_line = element.line_start
            else:
                # Add to current chunk
                current_chunk_elements.append(element)
                current_tokens += element_tokens
                
                # Check if we need to split due to token limit
                if current_tokens > self.config.max_tokens:
                    # Try to split at a good boundary
                    split_point = self._find_split_point(current_chunk_elements)
                    
                    if split_point > 0:
                        # Create chunk up to split point
                        chunk = self._create_chunk_from_elements(
                            current_chunk_elements[:split_point],
                            chunk_start_line
                        )
                        chunks.append(chunk)
                        
                        # Continue with remaining elements
                        current_chunk_elements = current_chunk_elements[split_point:]
                        current_tokens = sum(
                            self._estimate_tokens(e.content) 
                            for e in current_chunk_elements
                        )
                        chunk_start_line = current_chunk_elements[0].line_start
        
        # Create final chunk
        if current_chunk_elements:
            chunk = self._create_chunk_from_elements(
                current_chunk_elements,
                chunk_start_line
            )
            chunks.append(chunk)
        
        # Post-process chunks
        chunks = self.optimize_chunk_sizes(chunks)
        
        return chunks
    
    def chunk_by_markdown_structure(self, elements: List[MarkdownElement]) -> List[Chunk]:
        """
        Chunk content based on markdown structure
        
        Args:
            elements: List of markdown elements
            
        Returns:
            List of chunks
        """
        chunks = []
        current_section = []
        current_heading = None
        
        for element in elements:
            if element.element_type == MarkdownElementType.HEADING:
                # Start new section
                if current_section:
                    # Create chunk for previous section
                    chunk = self._create_section_chunk(
                        current_heading, current_section
                    )
                    chunks.append(chunk)
                
                current_heading = element
                current_section = []
            else:
                current_section.append(element)
                
                # Check if section is getting too large
                section_tokens = sum(
                    self._estimate_tokens(e.content) for e in current_section
                )
                if section_tokens > self.config.max_tokens:
                    # Split section
                    chunk = self._create_section_chunk(
                        current_heading, current_section
                    )
                    chunks.append(chunk)
                    current_section = []
        
        # Create final chunk
        if current_section or current_heading:
            chunk = self._create_section_chunk(current_heading, current_section)
            chunks.append(chunk)
        
        return chunks
    
    def optimize_chunk_sizes(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Optimize chunk sizes by merging small chunks and splitting large ones
        
        Args:
            chunks: List of chunks to optimize
            
        Returns:
            Optimized list of chunks
        """
        optimized = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Handle small chunks
            if current_chunk.token_count < self.config.min_tokens and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                
                # Try to merge with next chunk
                combined_tokens = current_chunk.token_count + next_chunk.token_count
                if combined_tokens <= self.config.max_tokens:
                    merged_chunk = self._merge_chunks(current_chunk, next_chunk)
                    optimized.append(merged_chunk)
                    i += 2  # Skip next chunk
                    continue
            
            # Handle large chunks
            elif current_chunk.token_count > self.config.max_tokens:
                split_chunks = self._split_large_chunk(current_chunk)
                optimized.extend(split_chunks)
                i += 1
                continue
            
            optimized.append(current_chunk)
            i += 1
        
        return optimized
    
    def preserve_code_integrity(self, content: str, elements: List[MarkdownElement]) -> List[Chunk]:
        """
        Create chunks while preserving code block integrity
        
        Args:
            content: Original content
            elements: List of markdown elements
            
        Returns:
            List of chunks with intact code blocks
        """
        chunks = []
        code_blocks = [e for e in elements if e.element_type == MarkdownElementType.CODE_BLOCK]
        
        for code_block in code_blocks:
            # Create dedicated chunk for each code block
            chunk = Chunk(
                chunk_id=self._generate_chunk_id(code_block.content),
                content=code_block.raw_content,
                chunk_type=ChunkType.CODE_BLOCK,
                line_start=code_block.line_start,
                line_end=code_block.line_end,
                token_count=self._estimate_tokens(code_block.content),
                elements=[code_block],
                metadata={
                    'language': code_block.language,
                    'preserved': True
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _should_split_at_element(self, 
                               element: MarkdownElement,
                               current_tokens: int,
                               element_tokens: int,
                               boundaries: List[SemanticBoundary]) -> bool:
        """Enhanced version that keeps tables intact"""
        
        # Never split before or after a table unless necessary
        if element.element_type == MarkdownElementType.TABLE:
            # Check if table fits in remaining space
            if current_tokens == 0:  # Start of chunk
                return False
            elif current_tokens + element_tokens <= self.config.max_tokens:
                return False  # Table fits, don't split
            else:
                return True  # Table doesn't fit, start new chunk
        
        # Always split at headings if configured
        if self.config.preserve_headings and element.element_type == MarkdownElementType.HEADING:
            return True
        
        # Check if we're at a semantic boundary
        for boundary in boundaries:
            if boundary.position == element.line_start and boundary.confidence > 0.8:
                return True
        
        # Check token limits
        if current_tokens + element_tokens > self.config.max_tokens:
            return True
        
        # Don't split code blocks if configured
        if self.config.preserve_code_blocks and element.element_type == MarkdownElementType.CODE_BLOCK:
            if element_tokens <= self.config.max_tokens:
                return current_tokens > 0  # Start new chunk if not empty
        
        return False
    
    def _find_split_point(self, elements: List[MarkdownElement]) -> int:
        """Find the best point to split a list of elements"""
        if len(elements) <= 1:
            return len(elements)
        
        # Try to find a natural break point
        total_tokens = 0
        for i, element in enumerate(elements):
            total_tokens += self._estimate_tokens(element.content)
            
            # Good split points:
            # - After a paragraph before a list
            # - After a list before a paragraph
            # - Between paragraphs
            if i < len(elements) - 1:
                current_type = element.element_type
                next_type = elements[i + 1].element_type
                
                if (current_type == MarkdownElementType.PARAGRAPH and 
                    next_type == MarkdownElementType.LIST_ITEM):
                    return i + 1
                elif (current_type == MarkdownElementType.LIST_ITEM and 
                      next_type == MarkdownElementType.PARAGRAPH):
                    return i + 1
                elif (current_type == MarkdownElementType.PARAGRAPH and 
                      next_type == MarkdownElementType.PARAGRAPH):
                    if total_tokens > self.config.max_tokens * 0.7:
                        return i + 1
        
        # Default: split in the middle
        return len(elements) // 2
    
    def _create_chunk_from_elements(self, 
                                  elements: List[MarkdownElement],
                                  start_line: int) -> Chunk:
        """Create a chunk from a list of elements"""
        if not elements:
            raise ValueError("Cannot create chunk from empty elements list")
        
        # Combine content
        content_parts = []
        for element in elements:
            content_parts.append(element.raw_content)
        
        content = '\n\n'.join(content_parts)
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(elements)
        
        # Calculate metrics
        token_count = self._estimate_tokens(content)
        end_line = elements[-1].line_end
        
        # Check for special content types
        metadata = self._extract_chunk_metadata(content, elements)
        
        chunk = Chunk(
            chunk_id=self._generate_chunk_id(content),
            content=content,
            chunk_type=chunk_type,
            line_start=start_line,
            line_end=end_line,
            token_count=token_count,
            elements=elements,
            metadata=metadata,
            quality_score=self._calculate_quality_score(elements, token_count),
            semantic_coherence=self._calculate_semantic_coherence(elements)
        )
        
        return chunk
    
    def _create_section_chunk(self, 
                            heading: Optional[MarkdownElement],
                            elements: List[MarkdownElement]) -> Chunk:
        """Create a chunk for a section (heading + content)"""
        all_elements = []
        if heading:
            all_elements.append(heading)
        all_elements.extend(elements)
        
        if not all_elements:
            raise ValueError("Cannot create empty section chunk")
        
        return self._create_chunk_from_elements(
            all_elements,
            all_elements[0].line_start
        )
    
    def _merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """Merge two chunks into one"""
        merged_content = f"{chunk1.content}\n\n{chunk2.content}"
        merged_elements = chunk1.elements + chunk2.elements
        
        merged_chunk = Chunk(
            chunk_id=self._generate_chunk_id(merged_content),
            content=merged_content,
            chunk_type=self._determine_chunk_type(merged_elements),
            line_start=chunk1.line_start,
            line_end=chunk2.line_end,
            token_count=chunk1.token_count + chunk2.token_count,
            elements=merged_elements,
            metadata={**chunk1.metadata, **chunk2.metadata, 'merged': True},
            quality_score=(chunk1.quality_score + chunk2.quality_score) / 2,
            semantic_coherence=(chunk1.semantic_coherence + chunk2.semantic_coherence) / 2
        )
        
        # Update parent-child relationships
        chunk1.child_chunk_ids.append(merged_chunk.chunk_id)
        chunk2.child_chunk_ids.append(merged_chunk.chunk_id)
        merged_chunk.parent_chunk_id = f"{chunk1.chunk_id},{chunk2.chunk_id}"
        
        return merged_chunk
    
    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split a large chunk into smaller ones"""
        if len(chunk.elements) == 1:
            # Single element chunk - split by content
            return self._split_single_element_chunk(chunk)
        
        # Multi-element chunk - split by elements
        split_point = self._find_split_point(chunk.elements)
        
        chunk1 = self._create_chunk_from_elements(
            chunk.elements[:split_point],
            chunk.line_start
        )
        chunk2 = self._create_chunk_from_elements(
            chunk.elements[split_point:],
            chunk.elements[split_point].line_start
        )
        
        # Update relationships
        chunk1.parent_chunk_id = chunk.chunk_id
        chunk2.parent_chunk_id = chunk.chunk_id
        chunk.child_chunk_ids = [chunk1.chunk_id, chunk2.chunk_id]
        
        # Recursively split if still too large
        result = []
        for sub_chunk in [chunk1, chunk2]:
            if sub_chunk.token_count > self.config.max_tokens:
                result.extend(self._split_large_chunk(sub_chunk))
            else:
                result.append(sub_chunk)
        
        return result
    
    def _split_single_element_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split a chunk containing a single large element"""
        element = chunk.elements[0]
        content = element.content
        
        # Split content into sentences or paragraphs
        if element.element_type == MarkdownElementType.PARAGRAPH:
            parts = self._split_paragraph(content, self.config.max_tokens)
        elif element.element_type == MarkdownElementType.CODE_BLOCK:
            parts = self._split_code_block(content, self.config.max_tokens)
        else:
            # Generic split
            parts = self._split_by_tokens(content, self.config.max_tokens)
        
        # Create chunks from parts
        chunks = []
        for i, part in enumerate(parts):
            sub_element = MarkdownElement(
                element_type=element.element_type,
                content=part,
                raw_content=part,
                line_start=element.line_start,
                line_end=element.line_end,
                level=element.level,
                language=element.language,
                metadata={**element.metadata, 'split_part': i + 1, 'total_parts': len(parts)}
            )
            
            sub_chunk = Chunk(
                chunk_id=self._generate_chunk_id(part),
                content=part,
                chunk_type=chunk.chunk_type,
                line_start=element.line_start,
                line_end=element.line_end,
                token_count=self._estimate_tokens(part),
                elements=[sub_element],
                metadata={
                    **chunk.metadata,
                    'split_from': chunk.chunk_id,
                    'part': i + 1,
                    'total_parts': len(parts)
                },
                parent_chunk_id=chunk.chunk_id
            )
            chunks.append(sub_chunk)
        
        return chunks
    
    def _split_paragraph(self, text: str, max_tokens: int) -> List[str]:
        """Split paragraph text into smaller parts"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        parts = []
        current_part = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens and current_part:
                parts.append(' '.join(current_part))
                current_part = [sentence]
                current_tokens = sentence_tokens
            else:
                current_part.append(sentence)
                current_tokens += sentence_tokens
        
        if current_part:
            parts.append(' '.join(current_part))
        
        return parts
    
    def _split_code_block(self, code: str, max_tokens: int) -> List[str]:
        """Split code block into smaller parts"""
        lines = code.split('\n')
        
        parts = []
        current_part = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self._estimate_tokens(line)
            
            if current_tokens + line_tokens > max_tokens and current_part:
                parts.append('\n'.join(current_part))
                current_part = [line]
                current_tokens = line_tokens
            else:
                current_part.append(line)
                current_tokens += line_tokens
        
        if current_part:
            parts.append('\n'.join(current_part))
        
        return parts
    
    def _split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Generic token-based splitting"""
        words = text.split()
        
        parts = []
        current_part = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self._estimate_tokens(word)
            
            if current_tokens + word_tokens > max_tokens and current_part:
                parts.append(' '.join(current_part))
                current_part = [word]
                current_tokens = word_tokens
            else:
                current_part.append(word)
                current_tokens += word_tokens
        
        if current_part:
            parts.append(' '.join(current_part))
        
        return parts
    
    def _split_paragraph_enhanced(self, text: str, max_tokens: int) -> List[str]:
        """Enhanced paragraph splitting with better sentence detection"""
        # Use the enhanced sentence splitter
        sentences = self.sentence_splitter.split_sentences(text)
        
        if not sentences:
            return [text]
        
        parts = []
        current_part = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If single sentence exceeds max tokens, split by words
            if sentence_tokens > max_tokens:
                if current_part:
                    parts.append(' '.join(current_part))
                    current_part = []
                
                # Split long sentence by clauses or words
                sub_parts = self._split_long_sentence(sentence, max_tokens)
                parts.extend(sub_parts)
                current_tokens = 0
                continue
            
            if current_tokens + sentence_tokens > max_tokens and current_part:
                parts.append(' '.join(current_part))
                current_part = [sentence]
                current_tokens = sentence_tokens
            else:
                current_part.append(sentence)
                current_tokens += sentence_tokens
        
        if current_part:
            parts.append(' '.join(current_part))
        
        return parts
    
    def _split_long_sentence(self, sentence: str, max_tokens: int) -> List[str]:
        """Split a long sentence intelligently"""
        # Try to split on clauses first
        clause_patterns = [
            r',\s+(?:and|but|or|nor|for|yet|so)\s+',  # Coordinating conjunctions
            r';\s+',  # Semicolons
            r',\s+(?:which|who|that|where|when)\s+',  # Relative clauses
            r',\s+',  # General comma splits
        ]
        
        for pattern in clause_patterns:
            parts = re.split(f'({pattern})', sentence)
            if len(parts) > 1:
                # Reconstruct with delimiters
                reconstructed = []
                for i in range(0, len(parts), 2):
                    if i + 1 < len(parts):
                        reconstructed.append(parts[i] + parts[i + 1])
                    else:
                        reconstructed.append(parts[i])
                
                # Check if this gives us reasonable chunks
                if all(self._estimate_tokens(p) <= max_tokens for p in reconstructed):
                    return reconstructed
        
        # Fall back to word-based splitting
        return self._split_by_tokens(sentence, max_tokens)
    
    def optimize_chunk_sizes_with_tables(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optimize chunk sizes while preserving tables"""
        optimized = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Never merge or split table chunks
            if current_chunk.chunk_type == ChunkType.TABLE:
                optimized.append(current_chunk)
                i += 1
                continue
            
            # Don't merge into table chunks
            if (i + 1 < len(chunks) and 
                chunks[i + 1].chunk_type == ChunkType.TABLE):
                optimized.append(current_chunk)
                i += 1
                continue
            
            # Otherwise, use standard optimization
            if current_chunk.token_count < self.config.min_tokens and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                
                # Skip if next is a table
                if next_chunk.chunk_type == ChunkType.TABLE:
                    optimized.append(current_chunk)
                    i += 1
                    continue
                
                # Try to merge with next chunk
                combined_tokens = current_chunk.token_count + next_chunk.token_count
                if combined_tokens <= self.config.max_tokens:
                    merged_chunk = self._merge_chunks(current_chunk, next_chunk)
                    optimized.append(merged_chunk)
                    i += 2
                    continue
            
            optimized.append(current_chunk)
            i += 1
        
        return optimized
    
    def _determine_chunk_type(self, elements: List[MarkdownElement]) -> ChunkType:
        """Determine the type of chunk based on its elements"""
        if not elements:
            return ChunkType.MIXED
        
        element_types = set(e.element_type for e in elements)
        
        # Single type chunks
        if len(element_types) == 1:
            single_type = elements[0].element_type
            if single_type == MarkdownElementType.HEADING:
                return ChunkType.HEADING
            elif single_type == MarkdownElementType.PARAGRAPH:
                return ChunkType.PARAGRAPH
            elif single_type == MarkdownElementType.CODE_BLOCK:
                return ChunkType.CODE_BLOCK
            elif single_type == MarkdownElementType.LIST_ITEM:
                return ChunkType.LIST_SECTION
        
        # Mixed type chunks - determine primary type
        type_counts = {}
        for element in elements:
            element_type = element.element_type
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
        
        # If heading is present, it's likely a section
        if MarkdownElementType.HEADING in element_types:
            return ChunkType.MIXED
        
        # Otherwise, use the most common type
        most_common = max(type_counts.items(), key=lambda x: x[1])[0]
        if most_common == MarkdownElementType.PARAGRAPH:
            return ChunkType.PARAGRAPH
        elif most_common == MarkdownElementType.CODE_BLOCK:
            return ChunkType.CODE_BLOCK
        elif most_common == MarkdownElementType.LIST_ITEM:
            return ChunkType.LIST_SECTION
        
        return ChunkType.MIXED
    
    def _extract_chunk_metadata(self, content: str, elements: List[MarkdownElement]) -> Dict[str, Any]:
        """Extract metadata about the chunk"""
        metadata = {
            'element_count': len(elements),
            'element_types': list(set(e.element_type.value for e in elements)),
            'has_code': any(e.element_type == MarkdownElementType.CODE_BLOCK for e in elements),
            'has_lists': any(e.element_type == MarkdownElementType.LIST_ITEM for e in elements),
        }
        
        # Check for special content patterns
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(content):
                metadata[f'contains_{pattern_name}'] = True
        
        # Extract languages from code blocks
        languages = set()
        for element in elements:
            if element.element_type == MarkdownElementType.CODE_BLOCK and element.language:
                languages.add(element.language)
        
        if languages:
            metadata['code_languages'] = list(languages)
        
        return metadata
    
    def _calculate_quality_score(self, elements: List[MarkdownElement], token_count: int) -> float:
        """Calculate quality score for a chunk"""
        score = 0.0
        
        # Token count score (prefer chunks close to optimal size)
        optimal_tokens = (self.config.max_tokens + self.config.min_tokens) / 2
        token_score = 1.0 - abs(token_count - optimal_tokens) / optimal_tokens
        score += token_score * 0.3
        
        # Element diversity score
        element_types = set(e.element_type for e in elements)
        diversity_score = min(len(element_types) / 3, 1.0)
        score += diversity_score * 0.2
        
        # Completeness score (no truncated elements)
        completeness_score = 1.0  # Simplified for now
        score += completeness_score * 0.3
        
        # Semantic coherence bonus
        if len(elements) > 1:
            # Check if elements are related (simplified)
            coherence_score = 0.8
        else:
            coherence_score = 1.0
        score += coherence_score * 0.2
        
        return min(score, 1.0)
    
    def _calculate_semantic_coherence(self, elements: List[MarkdownElement]) -> float:
        """Calculate how semantically coherent the elements are"""
        if len(elements) <= 1:
            return 1.0
        
        # Simple heuristic: elements of the same type are more coherent
        element_types = [e.element_type for e in elements]
        type_changes = sum(1 for i in range(1, len(element_types)) 
                          if element_types[i] != element_types[i-1])
        
        coherence = 1.0 - (type_changes / len(elements))
        
        # Bonus for sections with headings
        if any(e.element_type == MarkdownElementType.HEADING for e in elements):
            coherence = min(coherence + 0.1, 1.0)
        
        return coherence
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple approximation: 1 word ≈ 1.3 tokens
        word_count = len(text.split())
        return int(word_count * self.token_multiplier)
    
    def _generate_chunk_id(self, content: str) -> str:
        """Generate unique ID for chunk"""
        return f"chunk_{hashlib.md5(content.encode('utf-8')).hexdigest()[:12]}"
    
    def _hybrid_chunking(self, document_structure: DocumentStructure) -> List[Chunk]:
        """Hybrid approach combining semantic and structural chunking"""
        # First, chunk by structure
        structural_chunks = self.chunk_by_markdown_structure(document_structure.elements)
        
        # Then, refine using semantic boundaries
        refined_chunks = []
        for chunk in structural_chunks:
            if chunk.token_count > self.config.max_tokens:
                # Apply semantic chunking to large chunks
                sub_chunks = self.chunk_by_semantic_boundaries(
                    chunk.elements,
                    document_structure.semantic_boundaries
                )
                refined_chunks.extend(sub_chunks)
            else:
                refined_chunks.append(chunk)
        
        return refined_chunks
    
    def _fixed_size_chunking(self, elements: List[MarkdownElement]) -> List[Chunk]:
        """Simple fixed-size chunking (fallback method)"""
        chunks = []
        current_chunk_elements = []
        current_tokens = 0
        
        for element in elements:
            element_tokens = self._estimate_tokens(element.content)
            
            if current_tokens + element_tokens > self.config.max_tokens and current_chunk_elements:
                # Create chunk
                chunk = self._create_chunk_from_elements(
                    current_chunk_elements,
                    current_chunk_elements[0].line_start
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_elements = [element]
                current_tokens = element_tokens
            else:
                current_chunk_elements.append(element)
                current_tokens += element_tokens
        
        # Create final chunk
        if current_chunk_elements:
            chunk = self._create_chunk_from_elements(
                current_chunk_elements,
                current_chunk_elements[0].line_start
            )
            chunks.append(chunk)
        
        return chunks