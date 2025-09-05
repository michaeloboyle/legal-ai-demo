"""
Knowledge Distillation Service for Government Compliance
Agent: IlyaSutskever_AI & MLEngineering_Lead
GitHub Issues: #2, #7
"""

import asyncio
import torch
import torch.nn as nn
import hashlib
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

logger = logging.getLogger(__name__)

class KnowledgeDistillationService:
    """
    Production service for government document optimization using knowledge distillation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        
        # Initialize models lazily for faster startup
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
        self.sentence_transformer = None
        self.nlp = None
        
        # Cache for document optimizations
        self.optimization_cache = {}
        
        logger.info(f"KnowledgeDistillationService initialized on {self.device}")
    
    def default_config(self) -> Dict:
        """Default configuration for production deployment"""
        return {
            'teacher_model': 'microsoft/DialoGPT-large',  # Placeholder - would be custom legal model
            'student_model': 't5-base',
            'sentence_transformer': 'all-MiniLM-L6-v2',
            'max_length': 512,
            'target_grade_level': 10,
            'legal_accuracy_threshold': 0.90,
            'batch_size': 8,
            'temperature': 3.0,
            'num_beams': 3,
            'use_cache': True,
            'cache_ttl': 3600
        }
    
    async def load_models(self):
        """Load models asynchronously for production"""
        if self.models_loaded:
            return
        
        logger.info("Loading knowledge distillation models...")
        start_time = datetime.now()
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['student_model'])
            
            # Load student model (optimized for inference)
            self.student_model = AutoModel.from_pretrained(self.config['student_model'])
            self.student_model.to(self.device)
            self.student_model.eval()
            
            # Load sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer(self.config['sentence_transformer'])
            
            # Load spaCy for linguistic analysis
            self.nlp = spacy.load("en_core_web_sm")
            
            # Optional: Load quantized model for faster inference
            if hasattr(torch.quantization, 'quantize_dynamic'):
                self.student_model = torch.quantization.quantize_dynamic(
                    self.student_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            
            self.models_loaded = True
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Models loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    async def optimize_document(
        self,
        document: str,
        target_grade_level: int = 10,
        preserve_legal_accuracy: bool = True
    ) -> str:
        """
        Optimize government document for plain language compliance
        
        Args:
            document: Original government document text
            target_grade_level: Target reading grade level (6-12)
            preserve_legal_accuracy: Whether to preserve legal accuracy above threshold
            
        Returns:
            Optimized plain language version of the document
        """
        # Ensure models are loaded
        await self.load_models()
        
        # Check cache first
        doc_hash = self._hash_document(document, target_grade_level)
        if doc_hash in self.optimization_cache:
            logger.info("Cache hit for document optimization")
            return self.optimization_cache[doc_hash]
        
        start_time = datetime.now()
        
        try:
            # Preprocess document
            preprocessed = await self._preprocess_document(document)
            
            # Extract legal concepts that must be preserved
            legal_concepts = await self._extract_legal_concepts(preprocessed)
            
            # Apply knowledge distillation
            optimized_text = await self._apply_knowledge_distillation(
                document=preprocessed,
                legal_concepts=legal_concepts,
                target_grade_level=target_grade_level
            )
            
            # Validate legal accuracy if required
            if preserve_legal_accuracy:
                accuracy_score = await self._validate_legal_accuracy(document, optimized_text)
                if accuracy_score < self.config['legal_accuracy_threshold']:
                    logger.warning(f"Legal accuracy below threshold: {accuracy_score}")
                    # Apply conservative optimization
                    optimized_text = await self._conservative_optimization(
                        document, legal_concepts, target_grade_level
                    )
            
            # Post-process and validate
            final_text = await self._postprocess_document(optimized_text)
            
            # Cache result
            if self.config['use_cache']:
                self.optimization_cache[doc_hash] = final_text
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Document optimized in {processing_time:.2f} seconds")
            
            return final_text
            
        except Exception as e:
            logger.error(f"Error optimizing document: {str(e)}")
            raise
    
    async def _preprocess_document(self, document: str) -> Dict:
        """Preprocess government document for optimization"""
        # Clean and normalize text
        cleaned = self._clean_text(document)
        
        # Segment into sentences
        doc = self.nlp(cleaned)
        sentences = [sent.text for sent in doc.sents]
        
        # Identify document structure
        structure = self._identify_structure(cleaned)
        
        return {
            'original': document,
            'cleaned': cleaned,
            'sentences': sentences,
            'structure': structure,
            'token_count': len(doc)
        }
    
    async def _extract_legal_concepts(self, preprocessed_doc: Dict) -> Dict:
        """Extract legal concepts that must be preserved"""
        text = preprocessed_doc['cleaned']
        doc = self.nlp(text)
        
        legal_concepts = {
            'legal_entities': [],
            'regulatory_references': [],
            'legal_definitions': [],
            'citations': [],
            'requirements': []
        }
        
        # Extract named entities (organizations, laws, regulations)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'LAW', 'PERSON', 'GPE']:
                legal_concepts['legal_entities'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Extract regulatory references (CFR, USC, etc.)
        regulatory_patterns = [
            r'\b\d+ CFR \d+\b',
            r'\b\d+ U\.S\.C\. \d+\b',
            r'\bSection \d+\.\d+\b'
        ]
        
        import re
        for pattern in regulatory_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                legal_concepts['regulatory_references'].append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Extract legal definitions (terms in quotes or "means")
        definition_pattern = r'"([^"]+)"\s+means|shall\s+mean'
        matches = re.finditer(definition_pattern, text, re.IGNORECASE)
        for match in matches:
            legal_concepts['legal_definitions'].append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        return legal_concepts
    
    async def _apply_knowledge_distillation(
        self,
        document: Dict,
        legal_concepts: Dict,
        target_grade_level: int
    ) -> str:
        """Apply knowledge distillation with legal concept preservation"""
        
        sentences = document['sentences']
        optimized_sentences = []
        
        for sentence in sentences:
            # Check if sentence contains legal concepts
            has_legal_concepts = self._contains_legal_concepts(sentence, legal_concepts)
            
            if has_legal_concepts:
                # Conservative optimization for legal content
                optimized = await self._optimize_with_constraints(
                    sentence, legal_concepts, target_grade_level
                )
            else:
                # Aggressive optimization for general content
                optimized = await self._optimize_sentence(sentence, target_grade_level)
            
            optimized_sentences.append(optimized)
        
        return ' '.join(optimized_sentences)
    
    async def _optimize_sentence(self, sentence: str, target_grade_level: int) -> str:
        """Optimize individual sentence for readability"""
        
        # Tokenize input
        inputs = self.tokenizer(
            sentence,
            max_length=self.config['max_length'],
            truncation=True,
            return_tensors='pt',
            padding=True
        ).to(self.device)
        
        try:
            with torch.no_grad():
                # Generate simplified version
                output_ids = self.student_model.generate(
                    **inputs,
                    max_length=min(len(sentence.split()) * 2, self.config['max_length']),
                    num_beams=self.config['num_beams'],
                    temperature=1.0,
                    do_sample=False,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                
                # Decode output
                optimized = self.tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                )
                
                # Validate reading level
                grade_level = self._calculate_flesch_kincaid(optimized)
                
                if grade_level <= target_grade_level:
                    return optimized.strip()
                else:
                    # Apply additional simplification
                    return await self._further_simplify(optimized, target_grade_level)
                    
        except Exception as e:
            logger.warning(f"Error optimizing sentence, returning original: {str(e)}")
            return sentence
    
    async def _optimize_with_constraints(
        self,
        sentence: str,
        legal_concepts: Dict,
        target_grade_level: int
    ) -> str:
        """Conservative optimization preserving legal concepts"""
        
        # Identify legal terms in sentence
        protected_terms = self._identify_protected_terms(sentence, legal_concepts)
        
        # Replace complex non-legal terms only
        simplified = sentence
        
        # Simple word replacements for common legal language
        replacements = {
            'aforementioned': 'mentioned above',
            'shall': 'must',
            'pursuant to': 'under',
            'heretofore': 'before',
            'hereafter': 'after',
            'whereas': 'because',
            'provided that': 'if',
            'notwithstanding': 'despite'
        }
        
        for complex_term, simple_term in replacements.items():
            if complex_term.lower() in sentence.lower():
                # Only replace if not a protected legal term
                if not self._is_protected_term(complex_term, protected_terms):
                    simplified = simplified.replace(complex_term, simple_term)
        
        return simplified
    
    async def _validate_legal_accuracy(self, original: str, optimized: str) -> float:
        """Validate legal accuracy using semantic similarity"""
        
        try:
            # Get embeddings
            original_embedding = self.sentence_transformer.encode([original])
            optimized_embedding = self.sentence_transformer.encode([optimized])
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(original_embedding, optimized_embedding)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating legal accuracy: {str(e)}")
            return 0.0
    
    async def _conservative_optimization(
        self,
        document: str,
        legal_concepts: Dict,
        target_grade_level: int
    ) -> str:
        """Apply conservative optimization when legal accuracy is at risk"""
        
        # Apply only basic readability improvements
        sentences = document.split('.')
        optimized_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Basic improvements: sentence length, word choice
                improved = self._basic_readability_improvements(sentence.strip())
                optimized_sentences.append(improved)
        
        return '. '.join(optimized_sentences)
    
    def _basic_readability_improvements(self, sentence: str) -> str:
        """Apply basic readability improvements"""
        
        # Split long sentences
        if len(sentence.split()) > 20:
            # Try to split at conjunctions
            for conjunction in [', and ', ', but ', ', or ', '; ']:
                if conjunction in sentence:
                    parts = sentence.split(conjunction, 1)
                    if len(parts) == 2:
                        return f"{parts[0].strip()}. {parts[1].strip()}"
        
        # Simple word replacements
        replacements = {
            'utilize': 'use',
            'commence': 'start',
            'terminate': 'end',
            'assistance': 'help',
            'demonstrate': 'show',
            'sufficient': 'enough',
            'additional': 'more'
        }
        
        improved = sentence
        for complex_word, simple_word in replacements.items():
            improved = improved.replace(complex_word, simple_word)
        
        return improved
    
    def _calculate_flesch_kincaid(self, text: str) -> float:
        """Calculate Flesch-Kincaid grade level"""
        
        doc = self.nlp(text)
        
        # Count sentences, words, and syllables
        sentences = len(list(doc.sents))
        words = len([token for token in doc if not token.is_punct and not token.is_space])
        syllables = sum(self._count_syllables(token.text) for token in doc if token.is_alpha)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Flesch-Kincaid formula
        grade_level = (
            0.39 * (words / sentences) +
            11.8 * (syllables / words) -
            15.59
        )
        
        return max(0, grade_level)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        syllables = 0
        vowels = 'aeiouy'
        
        if word[0] in vowels:
            syllables += 1
        
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                syllables += 1
        
        if word.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    def _contains_legal_concepts(self, sentence: str, legal_concepts: Dict) -> bool:
        """Check if sentence contains legal concepts"""
        sentence_lower = sentence.lower()
        
        for concept_list in legal_concepts.values():
            for concept in concept_list:
                if concept['text'].lower() in sentence_lower:
                    return True
        
        return False
    
    def _identify_protected_terms(self, sentence: str, legal_concepts: Dict) -> List[str]:
        """Identify terms that should not be modified"""
        protected = []
        
        for concept_list in legal_concepts.values():
            for concept in concept_list:
                if concept['text'].lower() in sentence.lower():
                    protected.append(concept['text'])
        
        return protected
    
    def _is_protected_term(self, term: str, protected_terms: List[str]) -> bool:
        """Check if term is protected from modification"""
        return any(term.lower() in protected.lower() for protected in protected_terms)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common formatting issues
        cleaned = cleaned.replace('  ', ' ')
        cleaned = cleaned.replace('\n', ' ')
        
        return cleaned
    
    def _identify_structure(self, text: str) -> Dict:
        """Identify document structure"""
        structure = {
            'has_sections': False,
            'has_subsections': False,
            'has_lists': False,
            'has_definitions': False
        }
        
        # Simple structure detection
        if any(marker in text.lower() for marker in ['section', '§', 'subsection']):
            structure['has_sections'] = True
        
        if any(marker in text for marker in ['(a)', '(1)', '(i)']):
            structure['has_subsections'] = True
        
        if any(marker in text for marker in ['•', '*', '-', '1.', '2.']):
            structure['has_lists'] = True
        
        if 'means' in text.lower() or '"' in text:
            structure['has_definitions'] = True
        
        return structure
    
    async def _postprocess_document(self, text: str) -> str:
        """Post-process optimized document"""
        # Fix punctuation and capitalization
        sentences = text.split('.')
        processed_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Capitalize first letter
                processed = sentence.strip()
                if processed:
                    processed = processed[0].upper() + processed[1:]
                processed_sentences.append(processed)
        
        return '. '.join(processed_sentences)
    
    async def _further_simplify(self, text: str, target_grade_level: int) -> str:
        """Apply additional simplification if needed"""
        # Split long sentences
        sentences = text.split('.')
        simplified_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                words = sentence.split()
                if len(words) > 15:
                    # Try to split at halfway point near a conjunction
                    midpoint = len(words) // 2
                    for i in range(midpoint - 2, midpoint + 3):
                        if i < len(words) and words[i].lower() in ['and', 'or', 'but', 'because']:
                            part1 = ' '.join(words[:i])
                            part2 = ' '.join(words[i+1:])
                            simplified_sentences.extend([part1, part2])
                            break
                    else:
                        simplified_sentences.append(sentence.strip())
                else:
                    simplified_sentences.append(sentence.strip())
        
        return '. '.join(simplified_sentences)
    
    def _hash_document(self, document: str, target_grade_level: int) -> str:
        """Create hash for document caching"""
        content = f"{document}_{target_grade_level}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def health_check(self) -> str:
        """Health check for the service"""
        if self.models_loaded:
            return "healthy"
        else:
            return "initializing"
    
    async def get_model_info(self) -> Dict:
        """Get model information and statistics"""
        await self.load_models()
        
        return {
            'student_model': self.config['student_model'],
            'device': str(self.device),
            'models_loaded': self.models_loaded,
            'cache_size': len(self.optimization_cache),
            'config': self.config
        }