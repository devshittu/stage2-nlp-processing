"""
document_processor.py

Utility for processing Stage 1 documents and extracting fields according to
configuration mapping.

Features:
- Flexible field extraction with fallbacks
- Context field aggregation for prompts
- Field preservation for pass-through
- Text preprocessing and validation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.schemas.data_models import Stage1Document
from src.utils.config_manager import get_settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents from Stage 1 according to field mapping configuration.

    Extracts text, context, and metadata based on configured mappings.
    """

    def __init__(self):
        """Initialize document processor with configuration."""
        self.settings = get_settings()
        self.field_mapping = self.settings.document_field_mapping

    def extract_text(self, document: Stage1Document) -> str:
        """
        Extract primary text field from document with fallback logic.

        Args:
            document: Stage 1 document

        Returns:
            Extracted text

        Raises:
            ValueError: If no text field found
        """
        # Try primary text field
        primary_field = self.field_mapping.text_field
        text = getattr(document, primary_field, None)

        if text:
            logger.debug(f"Extracted text from primary field: {primary_field}")
            return text

        # Try fallback fields
        for fallback_field in self.field_mapping.text_field_fallbacks:
            text = getattr(document, fallback_field, None)
            if text:
                logger.info(
                    f"Extracted text from fallback field: {fallback_field}",
                    extra={"document_id": document.document_id}
                )
                return text

        # No text found
        raise ValueError(
            f"No text found in document {document.document_id}. "
            f"Tried fields: {primary_field}, {self.field_mapping.text_field_fallbacks}"
        )

    def extract_context_fields(self, document: Stage1Document) -> Dict[str, Any]:
        """
        Extract context fields for enhanced processing (used in prompts).

        Args:
            document: Stage 1 document

        Returns:
            Dictionary of context fields
        """
        context = {}

        for field_name in self.field_mapping.context_fields:
            value = getattr(document, field_name, None)
            if value:
                context[field_name] = value

        logger.debug(
            f"Extracted {len(context)} context fields",
            extra={"document_id": document.document_id, "fields": list(context.keys())}
        )

        return context

    def build_context_string(self, context: Dict[str, Any]) -> str:
        """
        Build formatted context string for prompts.

        Args:
            context: Context fields dictionary

        Returns:
            Formatted context string
        """
        parts = []

        # Add title
        if "cleaned_title" in context:
            parts.append(f"Title: {context['cleaned_title']}")

        # Add author
        if "cleaned_author" in context:
            parts.append(f"Author: {context['cleaned_author']}")

        # Add publication date
        if "cleaned_publication_date" in context:
            parts.append(f"Published: {context['cleaned_publication_date']}")

        # Add excerpt
        if "cleaned_excerpt" in context:
            parts.append(f"Summary: {context['cleaned_excerpt']}")

        # Add categories
        if "cleaned_categories" in context and context["cleaned_categories"]:
            categories = ", ".join(context["cleaned_categories"])
            parts.append(f"Categories: {categories}")

        # Add tags
        if "cleaned_tags" in context and context["cleaned_tags"]:
            tags = ", ".join(context["cleaned_tags"][:5])  # Limit to 5 tags
            parts.append(f"Tags: {tags}")

        return "\n".join(parts)

    def extract_preserve_fields(self, document: Stage1Document) -> Dict[str, Any]:
        """
        Extract fields to preserve in output (pass-through).

        Args:
            document: Stage 1 document

        Returns:
            Dictionary of preserved fields
        """
        preserved = {}

        for field_name in self.field_mapping.preserve_in_output:
            value = getattr(document, field_name, None)
            if value is not None:
                preserved[field_name] = value

        return preserved

    def get_normalized_date(self, document: Stage1Document) -> Optional[str]:
        """
        Get normalized publication date.

        Args:
            document: Stage 1 document

        Returns:
            ISO 8601 formatted date string or None
        """
        # Try cleaned_publication_date first
        date_str = document.cleaned_publication_date

        # Try temporal_metadata as fallback
        if not date_str:
            date_str = document.temporal_metadata

        # Validate and normalize
        if date_str:
            try:
                # Attempt to parse and re-format
                if "T" in date_str:
                    # Already ISO format
                    return date_str
                else:
                    # Parse and convert
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    return dt.isoformat()
            except Exception as e:
                logger.warning(
                    f"Failed to parse date: {date_str}",
                    extra={"document_id": document.document_id, "error": str(e)}
                )
                return date_str  # Return as-is

        return None

    def process_document(self, document: Stage1Document) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Process document and extract all necessary information.

        Args:
            document: Stage 1 document

        Returns:
            Tuple of (text, context_fields, preserved_fields)

        Raises:
            ValueError: If required fields missing
        """
        logger.info(f"Processing document: {document.document_id}")

        # Extract text
        text = self.extract_text(document)

        # Validate text length
        max_length = self.settings.general.max_text_length
        if len(text) > max_length:
            logger.warning(
                f"Document text exceeds max length ({len(text)} > {max_length}). Truncating.",
                extra={"document_id": document.document_id}
            )
            text = text[:max_length]

        # Extract context
        context_fields = self.extract_context_fields(document)

        # Extract preserved fields
        preserved_fields = self.extract_preserve_fields(document)

        # Add normalized date to preserved fields
        normalized_date = self.get_normalized_date(document)
        if normalized_date:
            preserved_fields["normalized_date"] = normalized_date

        logger.info(
            f"Document processed successfully",
            extra={
                "document_id": document.document_id,
                "text_length": len(text),
                "context_fields_count": len(context_fields),
                "preserved_fields_count": len(preserved_fields)
            }
        )

        return text, context_fields, preserved_fields

    def validate_document(self, document: Stage1Document) -> Tuple[bool, Optional[str]]:
        """
        Validate document has required fields.

        Args:
            document: Stage 1 document

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check document_id
        if not document.document_id:
            return False, "Missing document_id"

        # Check text availability
        try:
            self.extract_text(document)
        except ValueError as e:
            return False, str(e)

        # Additional validation can be added here

        return True, None


# =============================================================================
# Convenience Functions
# =============================================================================

_processor_instance: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """
    Get singleton document processor instance.

    Returns:
        DocumentProcessor instance
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DocumentProcessor()
    return _processor_instance


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test with sample document
    sample_doc = Stage1Document(
        document_id="test_doc_001",
        version="1.0",
        cleaned_text="This is a test article about technology and AI.",
        cleaned_title="Test Article",
        cleaned_author="John Doe",
        cleaned_publication_date="2024-01-15T10:30:00Z",
        cleaned_categories=["technology", "AI"],
        cleaned_tags=["machine learning", "neural networks"]
    )

    processor = get_document_processor()

    # Test validation
    is_valid, error = processor.validate_document(sample_doc)
    print(f"Document valid: {is_valid}, Error: {error}")

    # Test processing
    text, context, preserved = processor.process_document(sample_doc)
    print(f"\nExtracted text length: {len(text)}")
    print(f"Context fields: {list(context.keys())}")
    print(f"Preserved fields: {list(preserved.keys())}")

    # Test context string building
    context_str = processor.build_context_string(context)
    print(f"\nContext string:\n{context_str}")
