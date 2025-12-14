"""
dp_logic.py

Dependency Parsing (DP) service core logic for Stage 2 NLP Processing Service.
Extracts Subject-Object-Action (SOA) triplets and dependency relations from text using spaCy.

Features:
- GPU-accelerated dependency parsing using spaCy transformer models
- Subject-Object-Action (SOA) triplet extraction from dependency trees
- Comprehensive dependency relation extraction
- Batch processing support for efficient throughput
- Confidence scoring for triplets and relations
- Robust error handling and logging
- Singleton pattern for model management

Implementation Notes:
- Uses spaCy's en_core_web_trf model (transformer-based, GPU-optimized)
- Identifies subjects using nsubj, nsubjpass dependency relations
- Identifies objects using dobj, pobj, iobj relations
- Identifies actions as root verbs in sentences
- Calculates confidence scores based on dependency distance and parse structure
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import spacy
import torch
from spacy.tokens import Doc, Token

from src.schemas.data_models import (
    SOATriplet,
    DependencyRelation,
    EntitySpan
)
from src.utils.config_manager import get_settings, get_device
from src.utils.logger import PerformanceLogger, get_logger

logger = get_logger(__name__, service="dp_logic")


# =============================================================================
# Helper Data Structures
# =============================================================================

@dataclass
class ParsedToken:
    """Internal representation of a parsed token with metadata."""
    text: str
    lemma: str
    pos: str
    dep: str
    head: int
    start_char: int
    end_char: int
    children: List[int]


@dataclass
class TripletCandidate:
    """Internal representation of a potential SOA triplet."""
    subject_token: ParsedToken
    action_token: ParsedToken
    object_token: ParsedToken
    confidence: float
    sentence: str


# =============================================================================
# Dependency Parser Class
# =============================================================================

class DependencyParser:
    """
    Dependency parser for extracting SOA triplets and dependency relations.

    Uses spaCy's transformer-based model (en_core_web_trf) with GPU acceleration
    to parse dependency structures and extract Subject-Object-Action triplets.

    Attributes:
        model_name: spaCy model name
        nlp: Loaded spaCy model
        device: Device for inference (cuda or cpu)
        batch_size: Batch size for processing
        settings: Application settings

    Example:
        parser = DependencyParser()
        triplets, relations = parser.parse("The UK Government announced new policies.")
    """

    def __init__(self):
        """Initialize dependency parser with spaCy model and GPU configuration."""
        self.settings = get_settings()
        self.dp_settings = self.settings.dp_service
        self.model_name = self.dp_settings.model_name
        self.device = get_device()
        self.batch_size = self.dp_settings.batch_size
        self.nlp: Optional[spacy.Language] = None

        # Dependency relations to extract
        self.subject_deps = ["nsubj", "nsubjpass"]
        self.object_deps = ["dobj", "pobj", "iobj", "obj"]
        self.agent_deps = ["agent", "attr"]

        # Confidence thresholds
        self.min_triplet_confidence = self.dp_settings.min_triplet_confidence

        logger.info(
            "DependencyParser initialized",
            extra={
                "model_name": self.model_name,
                "device": self.device,
                "batch_size": self.batch_size,
                "gpu_enabled": self.settings.general.gpu_enabled
            }
        )

    def load_model(self) -> None:
        """
        Load spaCy model with GPU configuration.

        Raises:
            RuntimeError: If model loading fails
        """
        if self.nlp is not None:
            logger.debug("Model already loaded, skipping")
            return

        try:
            with PerformanceLogger("load_spacy_model", logger, model_name=self.model_name):
                logger.info(f"Loading spaCy model: {self.model_name}")

                # Load model
                self.nlp = spacy.load(self.model_name)

                # Configure GPU if available
                if self.device == "cuda" and torch.cuda.is_available():
                    spacy.require_gpu()
                    logger.info(
                        "GPU enabled for spaCy",
                        extra={
                            "cuda_version": torch.version.cuda,
                            "device_name": torch.cuda.get_device_name(0)
                        }
                    )
                else:
                    logger.info("Using CPU for spaCy inference")

                # Configure pipeline components
                # Disable unnecessary components for performance
                # Keep: tok2vec OR transformer (for trf models), tagger, parser, attribute_ruler, lemmatizer
                required_pipes = ["tok2vec", "transformer", "tagger", "parser", "attribute_ruler", "lemmatizer"]
                disabled_pipes = []
                for pipe_name in self.nlp.pipe_names:
                    if pipe_name not in required_pipes:
                        disabled_pipes.append(pipe_name)

                if disabled_pipes:
                    for pipe in disabled_pipes:
                        self.nlp.disable_pipe(pipe)
                    logger.debug(f"Disabled unnecessary pipes: {disabled_pipes}")

                logger.info(
                    "Model loaded successfully",
                    extra={
                        "model_name": self.model_name,
                        "active_pipes": self.nlp.pipe_names,
                        "vocab_size": len(self.nlp.vocab)
                    }
                )

        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load spaCy model '{self.model_name}': {e}")

    def parse(self, text: str) -> Tuple[List[SOATriplet], List[DependencyRelation]]:
        """
        Parse text to extract SOA triplets and dependency relations.

        Args:
            text: Input text to parse

        Returns:
            Tuple of (soa_triplets, dependency_relations)

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If parsing fails

        Example:
            triplets, relations = parser.parse("The government announced new policies.")
        """
        # Ensure model is loaded
        if self.nlp is None:
            self.load_model()

        # Validate input
        if not text or not text.strip():
            logger.warning("Empty text provided for parsing")
            return [], []

        try:
            with PerformanceLogger("parse_text", logger, text_length=len(text)):
                # Parse with spaCy
                doc = self.nlp(text)

                # Extract SOA triplets
                triplets = self._extract_soa_triplets(doc)

                # Extract dependency relations
                relations = self._extract_dependency_relations(doc)

                logger.info(
                    "Text parsed successfully",
                    extra={
                        "text_length": len(text),
                        "tokens": len(doc),
                        "sentences": len(list(doc.sents)),
                        "triplets_extracted": len(triplets),
                        "relations_extracted": len(relations)
                    }
                )

                return triplets, relations

        except Exception as e:
            logger.error(f"Failed to parse text: {e}", exc_info=True)
            raise RuntimeError(f"Dependency parsing failed: {e}")

    def parse_batch(self, texts: List[str]) -> List[Tuple[List[SOATriplet], List[DependencyRelation]]]:
        """
        Parse multiple texts in batch for improved throughput.

        Args:
            texts: List of input texts

        Returns:
            List of tuples (soa_triplets, dependency_relations) for each text

        Example:
            results = parser.parse_batch(["Text 1", "Text 2", "Text 3"])
            for triplets, relations in results:
                print(f"Found {len(triplets)} triplets")
        """
        # Ensure model is loaded
        if self.nlp is None:
            self.load_model()

        if not texts:
            logger.warning("Empty text list provided for batch parsing")
            return []

        try:
            with PerformanceLogger("parse_batch", logger, batch_size=len(texts)):
                results = []

                # Process texts in batch using spaCy's pipe
                docs = list(self.nlp.pipe(
                    texts,
                    batch_size=self.batch_size,
                    n_process=1  # Use single process for GPU
                ))

                # Extract triplets and relations for each document
                for i, doc in enumerate(docs):
                    try:
                        triplets = self._extract_soa_triplets(doc)
                        relations = self._extract_dependency_relations(doc)
                        results.append((triplets, relations))
                    except Exception as e:
                        logger.error(
                            f"Failed to extract from document {i}: {e}",
                            extra={"document_index": i}
                        )
                        # Return empty results for failed document
                        results.append(([], []))

                logger.info(
                    "Batch parsing completed",
                    extra={
                        "documents_processed": len(texts),
                        "total_triplets": sum(len(r[0]) for r in results),
                        "total_relations": sum(len(r[1]) for r in results)
                    }
                )

                return results

        except Exception as e:
            logger.error(f"Batch parsing failed: {e}", exc_info=True)
            raise RuntimeError(f"Batch parsing failed: {e}")

    def _extract_soa_triplets(self, doc: Doc) -> List[SOATriplet]:
        """
        Extract Subject-Object-Action triplets from dependency parse.

        Args:
            doc: spaCy Doc object

        Returns:
            List of SOATriplet objects
        """
        triplets = []

        # Process each sentence separately
        for sent in doc.sents:
            # Find candidates in sentence
            candidates = self._find_triplet_candidates(sent)

            # Convert candidates to SOATriplet objects
            for candidate in candidates:
                if candidate.confidence >= self.min_triplet_confidence:
                    try:
                        triplet = SOATriplet(
                            subject=EntitySpan(
                                text=candidate.subject_token.text,
                                start_char=candidate.subject_token.start_char,
                                end_char=candidate.subject_token.end_char
                            ),
                            action=EntitySpan(
                                text=candidate.action_token.text,
                                start_char=candidate.action_token.start_char,
                                end_char=candidate.action_token.end_char
                            ),
                            object=EntitySpan(
                                text=candidate.object_token.text,
                                start_char=candidate.object_token.start_char,
                                end_char=candidate.object_token.end_char
                            ),
                            confidence=candidate.confidence,
                            sentence=candidate.sentence
                        )
                        triplets.append(triplet)
                    except Exception as e:
                        logger.warning(
                            f"Failed to create SOATriplet: {e}",
                            extra={"sentence": candidate.sentence}
                        )

        return triplets

    def _find_triplet_candidates(self, sent: spacy.tokens.Span) -> List[TripletCandidate]:
        """
        Find SOA triplet candidates in a sentence.

        Args:
            sent: spaCy Span representing a sentence

        Returns:
            List of TripletCandidate objects
        """
        candidates = []

        # Find root verb (action)
        action_tokens = [token for token in sent if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]]

        logger.debug(f"Sentence: {sent.text}")
        logger.debug(f"Found {len(action_tokens)} ROOT verbs: {[t.text for t in action_tokens]}")

        for action in action_tokens:
            # Find subjects
            subjects = self._find_subjects(action)
            logger.debug(f"Action '{action.text}': found {len(subjects)} subjects: {[s.text for s in subjects]}")

            # Find objects
            objects = self._find_objects(action)
            logger.debug(f"Action '{action.text}': found {len(objects)} objects: {[o.text for o in objects]}")

            # Create candidates for each subject-object pair
            for subj in subjects:
                for obj in objects:
                    confidence = self._calculate_triplet_confidence(subj, action, obj)

                    candidate = TripletCandidate(
                        subject_token=self._token_to_parsed(subj),
                        action_token=self._token_to_parsed(action),
                        object_token=self._token_to_parsed(obj),
                        confidence=confidence,
                        sentence=sent.text
                    )
                    candidates.append(candidate)
                    logger.debug(f"Created candidate: {subj.text} -> {action.text} -> {obj.text} (confidence: {confidence:.2f})")

        logger.debug(f"Total candidates for sentence: {len(candidates)}")
        return candidates

    def _find_subjects(self, token: Token) -> List[Token]:
        """
        Find subject tokens related to a verb.

        Args:
            token: Verb token

        Returns:
            List of subject tokens
        """
        subjects = []

        for child in token.children:
            if child.dep_ in self.subject_deps:
                # Get the full noun phrase if available
                subjects.append(child)
            elif child.dep_ in self.agent_deps:
                # Handle passive constructions
                subjects.append(child)

        return subjects

    def _find_objects(self, token: Token) -> List[Token]:
        """
        Find object tokens related to a verb.

        Args:
            token: Verb token

        Returns:
            List of object tokens
        """
        objects = []

        for child in token.children:
            if child.dep_ in self.object_deps:
                objects.append(child)
            # Also check for prepositional objects
            elif child.dep_ == "prep":
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        objects.append(prep_child)

        return objects

    def _calculate_triplet_confidence(self, subject: Token, action: Token, obj: Token) -> float:
        """
        Calculate confidence score for a SOA triplet.

        Confidence is based on:
        - Dependency distance (closer is better)
        - POS tag confidence
        - Sentence structure

        Args:
            subject: Subject token
            action: Action token
            obj: Object token

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.8

        # Adjust for dependency distance
        subj_distance = abs(subject.i - action.i)
        obj_distance = abs(obj.i - action.i)
        max_distance = 10

        distance_penalty = (subj_distance + obj_distance) / (2 * max_distance)
        confidence -= min(distance_penalty * 0.2, 0.2)

        # Boost for direct dependencies
        if subject.head == action:
            confidence += 0.1
        if obj.head == action:
            confidence += 0.1

        # Ensure within bounds
        confidence = max(0.0, min(1.0, confidence))

        return confidence

    def _extract_dependency_relations(self, doc: Doc) -> List[DependencyRelation]:
        """
        Extract all dependency relations from parsed document.

        Args:
            doc: spaCy Doc object

        Returns:
            List of DependencyRelation objects
        """
        relations = []

        # Extract configured dependency relations
        target_relations = set(self.dp_settings.dependency_relations)

        for token in doc:
            # Skip punctuation and spaces
            if token.is_punct or token.is_space:
                continue

            # Check if this is a relation we want to extract
            if token.dep_ in target_relations or not target_relations:
                try:
                    relation = DependencyRelation(
                        head=token.head.text,
                        head_pos=token.head.pos_,
                        dependent=token.text,
                        dependent_pos=token.pos_,
                        relation=token.dep_,
                        confidence=1.0  # spaCy doesn't provide confidence scores
                    )
                    relations.append(relation)
                except Exception as e:
                    logger.warning(
                        f"Failed to create DependencyRelation: {e}",
                        extra={"token": token.text, "dep": token.dep_}
                    )

        return relations

    def _token_to_parsed(self, token: Token) -> ParsedToken:
        """
        Convert spaCy Token to ParsedToken.

        Args:
            token: spaCy Token

        Returns:
            ParsedToken object
        """
        return ParsedToken(
            text=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            dep=token.dep_,
            head=token.head.i,
            start_char=token.idx,
            end_char=token.idx + len(token.text),
            children=[child.i for child in token.children]
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded model.

        Returns:
            Dictionary with model information
        """
        if self.nlp is None:
            return {
                "model_name": self.model_name,
                "loaded": False
            }

        return {
            "model_name": self.model_name,
            "loaded": True,
            "device": self.device,
            "batch_size": self.batch_size,
            "active_pipes": self.nlp.pipe_names,
            "vocab_size": len(self.nlp.vocab),
            "gpu_enabled": self.device == "cuda"
        }


# =============================================================================
# Singleton Pattern
# =============================================================================

_parser_instance: Optional[DependencyParser] = None


def get_dependency_parser() -> DependencyParser:
    """
    Get singleton dependency parser instance.

    Returns:
        DependencyParser instance

    Example:
        parser = get_dependency_parser()
        triplets, relations = parser.parse("Some text")
    """
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = DependencyParser()
        # Eagerly load model
        _parser_instance.load_model()
    return _parser_instance


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_text(text: str) -> Tuple[List[SOATriplet], List[DependencyRelation]]:
    """
    Parse text and extract SOA triplets and dependencies (convenience function).

    Args:
        text: Input text

    Returns:
        Tuple of (soa_triplets, dependency_relations)
    """
    parser = get_dependency_parser()
    return parser.parse(text)


def parse_batch(texts: List[str]) -> List[Tuple[List[SOATriplet], List[DependencyRelation]]]:
    """
    Parse multiple texts in batch (convenience function).

    Args:
        texts: List of input texts

    Returns:
        List of tuples (soa_triplets, dependency_relations)
    """
    parser = get_dependency_parser()
    return parser.parse_batch(texts)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test text
    test_texts = [
        "The UK Government announced new policies yesterday.",
        "Apple released a new iPhone model in September.",
        "Scientists discovered a new species in the Amazon rainforest.",
        "The president signed an agreement with foreign leaders."
    ]

    try:
        # Initialize parser
        print("Initializing DependencyParser...")
        parser = get_dependency_parser()

        # Test single parse
        print("\n=== Testing Single Parse ===")
        text = test_texts[0]
        print(f"Text: {text}")

        triplets, relations = parser.parse(text)

        print(f"\nFound {len(triplets)} SOA triplets:")
        for i, triplet in enumerate(triplets):
            print(f"{i+1}. Subject: {triplet.subject.text}")
            print(f"   Action: {triplet.action.text}")
            print(f"   Object: {triplet.object.text}")
            print(f"   Confidence: {triplet.confidence:.2f}")

        print(f"\nFound {len(relations)} dependency relations:")
        for i, rel in enumerate(relations[:5]):  # Show first 5
            print(f"{i+1}. {rel.dependent} ({rel.dependent_pos}) --{rel.relation}--> {rel.head} ({rel.head_pos})")

        # Test batch parse
        print("\n=== Testing Batch Parse ===")
        print(f"Processing {len(test_texts)} texts...")

        results = parser.parse_batch(test_texts)

        for i, (triplets, relations) in enumerate(results):
            print(f"\nDocument {i+1}: {test_texts[i][:50]}...")
            print(f"  Triplets: {len(triplets)}, Relations: {len(relations)}")

        # Model info
        print("\n=== Model Information ===")
        info = parser.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")

        print("\n✓ All tests completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
