#!/usr/bin/env python3
"""
Enhanced Sentence Splitter for Enhanced Multi-Embedding Entry System
Provides robust sentence splitting that handles edge cases like abbreviations, decimals, and URLs
"""

import re
from typing import List


class SentenceSplitter:
    """Robust sentence splitter that handles edge cases"""
    
    def __init__(self):
        # Abbreviations that don't end sentences
        self.abbreviations = {
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 
            'Ph.D', 'M.D', 'B.A', 'M.A', 'D.D.S', 'B.S', 'M.S',
            'LL.B', 'LL.M', 'vs', 'etc', 'inc', 'ltd', 'co', 'corp',
            'eg', 'ie', 'cf', 'al', 'ed', 'eds', 'vol', 'no', 'p', 'pp',
            'para', 'i.e', 'e.g', 'viz', 'sc', 'ca', 'v', 'viz',
            'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Sept',
            'Oct', 'Nov', 'Dec', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'
        }
        
        # Build regex pattern for abbreviations
        abbrev_pattern = '|'.join(re.escape(abbr) for abbr in self.abbreviations)
        self.abbrev_regex = re.compile(f'\\b({abbrev_pattern})\\.', re.IGNORECASE)
        
        # Sentence ending punctuation
        self.sentence_end = re.compile(r'([.!?])\s*')
        
        # Special patterns that should not be split
        self.no_split_patterns = [
            re.compile(r'\d+\.\d+'),  # Decimal numbers
            re.compile(r'\b\w\.\w\.'),  # U.S.A., etc.
            re.compile(r'https?://[^\s]+'),  # URLs
            re.compile(r'\w+\.\w+@\w+'),  # Email-like patterns
            re.compile(r'[A-Z]\.[A-Z]'),  # P.O. Box, etc.
        ]
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences handling edge cases"""
        if not text.strip():
            return []
        
        # Temporarily replace patterns we don't want to split
        replacements = []
        temp_text = text
        
        # Replace decimal numbers
        for match in re.finditer(r'\d+\.\d+', temp_text):
            placeholder = f"__DECIMAL{len(replacements)}__"
            replacements.append((placeholder, match.group()))
            temp_text = temp_text.replace(match.group(), placeholder)
        
        # Replace abbreviations
        for match in self.abbrev_regex.finditer(temp_text):
            placeholder = f"__ABBREV{len(replacements)}__"
            replacements.append((placeholder, match.group()))
            temp_text = temp_text[:match.start()] + placeholder + temp_text[match.end():]
        
        # Replace URLs
        for match in re.finditer(r'https?://[^\s]+', temp_text):
            placeholder = f"__URL{len(replacements)}__"
            replacements.append((placeholder, match.group()))
            temp_text = temp_text.replace(match.group(), placeholder)
        
        # Now split on sentence boundaries
        sentences = []
        current_sentence = []
        
        # Split by sentence-ending punctuation
        parts = self.sentence_end.split(temp_text)
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            if i + 1 < len(parts) and parts[i + 1] in '.!?':
                # This part ends with punctuation
                current_sentence.append(part + parts[i + 1])
                
                # Check if next part starts with lowercase (might be continuation)
                if i + 2 < len(parts) and parts[i + 2] and parts[i + 2][0].islower():
                    # Might be a false split
                    current_sentence.append(' ')
                    i += 2
                    continue
                else:
                    # End of sentence
                    sentence = ''.join(current_sentence).strip()
                    if sentence:
                        sentences.append(sentence)
                    current_sentence = []
                    i += 2
            else:
                # No punctuation, add to current sentence
                if part:
                    current_sentence.append(part)
                i += 1
        
        # Add any remaining content
        if current_sentence:
            sentence = ' '.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
        
        # Restore replacements
        restored_sentences = []
        for sentence in sentences:
            for placeholder, original in replacements:
                sentence = sentence.replace(placeholder, original)
            restored_sentences.append(sentence)
        
        # Post-process to merge incorrectly split sentences
        merged_sentences = self._merge_incorrectly_split(restored_sentences)
        
        return merged_sentences
    
    def _merge_incorrectly_split(self, sentences: List[str]) -> List[str]:
        """Merge sentences that were incorrectly split"""
        if len(sentences) <= 1:
            return sentences
        
        merged = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            
            # Check if this sentence should be merged with the next
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1]
                
                # Merge if current ends with abbreviation and next starts with lowercase
                if (self._ends_with_abbreviation(current) and 
                    next_sentence and next_sentence[0].islower()):
                    merged.append(current + ' ' + next_sentence)
                    i += 2
                    continue
                
                # Merge if current is too short (likely a fragment)
                if len(current.split()) < 3 and not current.endswith(('!', '?')):
                    merged.append(current + ' ' + next_sentence)
                    i += 2
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _ends_with_abbreviation(self, sentence: str) -> bool:
        """Check if sentence ends with an abbreviation"""
        words = sentence.split()
        if not words:
            return False
        
        last_word = words[-1]
        # Remove trailing period for comparison
        if last_word.endswith('.'):
            word_without_period = last_word[:-1]
            return word_without_period in self.abbreviations or word_without_period.lower() in [a.lower() for a in self.abbreviations]
        
        return False