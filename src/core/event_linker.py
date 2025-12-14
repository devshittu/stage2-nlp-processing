"""
event_linker.py

Sophisticated event linking and storyline distinction system.

Implements multi-dimensional event similarity analysis to prevent conflation of distinct
storylines (e.g., separating Trump+Israel/Gaza from Trump+Qatar economic partnership).

Key Features:
- Multi-dimensional similarity (semantic, entity, temporal, domain)
- Entity-role-context disambiguation
- Domain-aware clustering
- Causality detection and chain construction
- Hierarchical agglomerative clustering for storyline identification

Algorithm:
1. Compute pairwise event similarities across multiple dimensions
2. Apply domain boundaries to prevent cross-domain linking
3. Perform entity-role-context analysis for disambiguation
4. Cluster events into storylines using weighted distance metric
5. Identify key events and causal chains within storylines
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config_manager import get_settings
from src.utils.logger import get_logger, PerformanceLogger
from src.schemas.data_models import (
    Event, EventLinkage, Storyline, create_storyline_id
)

logger = get_logger(__name__)


# =============================================================================
# Event Linker
# =============================================================================

class EventLinker:
    """
    Links related events and identifies distinct storylines.

    Uses multi-dimensional similarity analysis to distinguish between
    different narratives involving similar entities.
    """

    def __init__(self):
        """Initialize event linker with embedding model."""
        self.settings = get_settings().event_linking
        self.general_settings = get_settings().general

        # Load embedding model
        logger.info(f"Loading embedding model: {self.settings.embedding_model}")
        self.embedding_model = SentenceTransformer(self.settings.embedding_model)

        # Try to use GPU if enabled and available
        if self.general_settings.gpu_enabled:
            try:
                import torch
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.cuda()
                    logger.info("Event linker using GPU for embeddings")
                else:
                    logger.warning("GPU enabled in config but CUDA not available, using CPU")
            except Exception as e:
                logger.warning(f"Failed to move embedding model to GPU: {e}, using CPU")

        logger.info("Event linker initialized successfully")

    def link_events(
        self,
        events: List[Event],
        batch_id: str = "batch"
    ) -> Tuple[List[EventLinkage], List[Storyline]]:
        """
        Link events and identify storylines.

        Args:
            events: List of events to link
            batch_id: Batch identifier

        Returns:
            Tuple of (linkages, storylines)
        """
        if len(events) < 2:
            logger.info("Less than 2 events, skipping linking")
            return [], []

        logger.info(f"Linking {len(events)} events")

        with PerformanceLogger("event_linking", logger.logger, event_count=len(events)):
            # Step 1: Compute embeddings
            embeddings = self._compute_embeddings(events)

            # Step 2: Build similarity matrix
            similarity_matrix = self._build_similarity_matrix(events, embeddings)

            # Step 3: Identify linkages
            linkages = self._identify_linkages(events, similarity_matrix)

            # Step 4: Cluster into storylines
            storylines = self._cluster_storylines(events, similarity_matrix, batch_id)

            # Step 5: Assign storyline IDs to events
            self._assign_storyline_ids(events, storylines)

            # Step 6: Detect causal chains
            self._detect_causal_chains(events, linkages, storylines)

            logger.info(
                f"Linked {len(linkages)} event pairs into {len(storylines)} storylines",
                extra={"batch_id": batch_id}
            )

            return linkages, storylines

    def _compute_embeddings(self, events: List[Event]) -> np.ndarray:
        """
        Compute embeddings for event descriptions.

        Args:
            events: List of events

        Returns:
            Embedding matrix (n_events x embedding_dim)
        """
        # Build event descriptions
        descriptions = []
        for event in events:
            desc_parts = [
                f"Event type: {event.event_type}",
                f"Trigger: {event.trigger.text}",
            ]

            # Add arguments
            for arg in event.arguments:
                desc_parts.append(f"{arg.argument_role}: {arg.entity.text}")

            # Add domain
            if event.domain:
                desc_parts.append(f"Domain: {event.domain}")

            # Add metadata
            if event.metadata.sentiment:
                desc_parts.append(f"Sentiment: {event.metadata.sentiment}")

            descriptions.append(" | ".join(desc_parts))

        # Compute embeddings
        logger.debug(f"Computing embeddings for {len(descriptions)} events")
        embeddings = self.embedding_model.encode(
            descriptions,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )

        return embeddings

    def _build_similarity_matrix(
        self,
        events: List[Event],
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Build multi-dimensional similarity matrix.

        Combines:
        - Semantic similarity (embeddings)
        - Entity overlap (Jaccard)
        - Temporal proximity
        - Domain similarity

        Args:
            events: List of events
            embeddings: Event embeddings

        Returns:
            Similarity matrix (n_events x n_events)
        """
        n_events = len(events)
        weights = self.settings.storyline_clustering.weights

        # 1. Semantic similarity
        semantic_sim = cosine_similarity(embeddings)

        # 2. Entity overlap similarity
        entity_sim = np.zeros((n_events, n_events))
        for i in range(n_events):
            for j in range(i + 1, n_events):
                overlap = self._compute_entity_overlap(events[i], events[j])
                entity_sim[i, j] = overlap
                entity_sim[j, i] = overlap

        # 3. Temporal proximity
        temporal_sim = np.zeros((n_events, n_events))
        for i in range(n_events):
            for j in range(i + 1, n_events):
                proximity = self._compute_temporal_proximity(events[i], events[j])
                temporal_sim[i, j] = proximity
                temporal_sim[j, i] = proximity

        # 4. Domain similarity
        domain_sim = np.zeros((n_events, n_events))
        for i in range(n_events):
            for j in range(i + 1, n_events):
                same_domain = self._compute_domain_similarity(events[i], events[j])
                domain_sim[i, j] = same_domain
                domain_sim[j, i] = same_domain

        # Combine with weights
        combined_sim = (
            weights.semantic * semantic_sim +
            weights.entity * entity_sim +
            weights.temporal * temporal_sim +
            weights.domain * domain_sim
        )

        # Apply domain boundaries if enforced
        if self.settings.enforce_domain_boundaries and not self.settings.allow_cross_domain_linking:
            for i in range(n_events):
                for j in range(i + 1, n_events):
                    if not self._can_link_domains(events[i], events[j]):
                        combined_sim[i, j] = 0.0
                        combined_sim[j, i] = 0.0

        logger.debug("Built multi-dimensional similarity matrix")
        return combined_sim

    def _compute_entity_overlap(self, event1: Event, event2: Event) -> float:
        """
        Compute entity overlap using Jaccard similarity with context awareness.

        This is critical for storyline distinction. We don't just compare entity names,
        but also consider their roles and contexts to prevent conflation.

        For example, "Trump" as agent in Israel context vs "Trump" as agent in
        Qatar context should have lower similarity.

        Args:
            event1: First event
            event2: Second event

        Returns:
            Jaccard similarity with context weighting (0.0 to 1.0)
        """
        # Extract entity-role-context triplets
        def get_entity_signatures(event: Event) -> Set[str]:
            signatures = set()
            for arg in event.arguments:
                # Create signature: entity_text|role|domain
                sig = f"{arg.entity.text.lower()}|{arg.argument_role}|{event.domain or 'unknown'}"

                # Add context if available and entity-role-context enabled
                if self.settings.enable_entity_role_context and arg.entity.context:
                    # Hash context to keep signature manageable
                    context_hash = hashlib.md5(arg.entity.context.encode()).hexdigest()[:8]
                    sig += f"|{context_hash}"

                signatures.add(sig)
            return signatures

        sigs1 = get_entity_signatures(event1)
        sigs2 = get_entity_signatures(event2)

        if not sigs1 or not sigs2:
            return 0.0

        # Jaccard similarity
        intersection = len(sigs1 & sigs2)
        union = len(sigs1 | sigs2)

        return intersection / union if union > 0 else 0.0

    def _compute_temporal_proximity(self, event1: Event, event2: Event) -> float:
        """
        Compute temporal proximity (1.0 = same time, 0.0 = far apart).

        Args:
            event1: First event
            event2: Second event

        Returns:
            Proximity score (0.0 to 1.0)
        """
        # Extract temporal references
        time1 = event1.temporal_reference
        time2 = event2.temporal_reference

        if not time1 or not time2:
            # No temporal info, assume moderate proximity
            return 0.5

        try:
            # Parse timestamps
            dt1 = datetime.fromisoformat(time1.replace("Z", "+00:00"))
            dt2 = datetime.fromisoformat(time2.replace("Z", "+00:00"))

            # Compute difference in days
            diff_days = abs((dt2 - dt1).days)

            # Map to similarity score using exponential decay
            window_days = self.settings.temporal_window_days
            similarity = np.exp(-diff_days / window_days)

            return float(similarity)

        except Exception as e:
            logger.debug(f"Failed to parse temporal references: {e}")
            return 0.5

    def _compute_domain_similarity(self, event1: Event, event2: Event) -> float:
        """
        Compute domain similarity.

        Args:
            event1: First event
            event2: Second event

        Returns:
            1.0 if same domain, 0.0 otherwise
        """
        if not event1.domain or not event2.domain:
            return 0.5  # Unknown domains

        return 1.0 if event1.domain == event2.domain else 0.0

    def _can_link_domains(self, event1: Event, event2: Event) -> bool:
        """
        Check if two events can be linked across domains.

        Args:
            event1: First event
            event2: Second event

        Returns:
            True if linking allowed
        """
        if not self.settings.enforce_domain_boundaries:
            return True

        if not event1.domain or not event2.domain:
            return True  # Unknown domains can link

        return event1.domain == event2.domain

    def _identify_linkages(
        self,
        events: List[Event],
        similarity_matrix: np.ndarray
    ) -> List[EventLinkage]:
        """
        Identify event linkages above threshold.

        Args:
            events: List of events
            similarity_matrix: Similarity matrix

        Returns:
            List of event linkages
        """
        linkages = []
        threshold = self.settings.semantic_similarity_threshold

        n_events = len(events)

        for i in range(n_events):
            for j in range(i + 1, n_events):
                similarity = similarity_matrix[i, j]

                if similarity >= threshold:
                    linkage = EventLinkage(
                        source_event_id=events[i].event_id,
                        target_event_id=events[j].event_id,
                        link_type="coreference",  # Can be refined based on analysis
                        similarity_score=float(similarity),
                        semantic_similarity=float(cosine_similarity(
                            [self.embedding_model.encode(events[i].trigger.text)],
                            [self.embedding_model.encode(events[j].trigger.text)]
                        )[0, 0]),
                        entity_overlap=self._compute_entity_overlap(events[i], events[j]),
                        temporal_proximity=self._compute_temporal_proximity(events[i], events[j]),
                        domain_similarity=self._compute_domain_similarity(events[i], events[j])
                    )
                    linkages.append(linkage)

        logger.debug(f"Identified {len(linkages)} linkages above threshold {threshold}")
        return linkages

    def _cluster_storylines(
        self,
        events: List[Event],
        similarity_matrix: np.ndarray,
        batch_id: str
    ) -> List[Storyline]:
        """
        Cluster events into storylines using hierarchical clustering.

        Args:
            events: List of events
            similarity_matrix: Similarity matrix
            batch_id: Batch identifier

        Returns:
            List of storylines
        """
        if not self.settings.storyline_clustering.enabled:
            logger.info("Storyline clustering disabled")
            return []

        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix

        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=1.0 - self.settings.semantic_similarity_threshold
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        # Build storylines
        storylines = []
        cluster_to_events = defaultdict(list)

        for event, label in zip(events, cluster_labels):
            cluster_to_events[label].append(event)

        for cluster_idx, cluster_events in cluster_to_events.items():
            # Filter by min cluster size
            if len(cluster_events) < self.settings.storyline_clustering.min_cluster_size:
                continue

            storyline = self._build_storyline(
                cluster_events,
                cluster_idx,
                batch_id
            )
            storylines.append(storyline)

        logger.info(f"Identified {len(storylines)} storylines from {len(events)} events")
        return storylines

    def _build_storyline(
        self,
        events: List[Event],
        cluster_idx: int,
        batch_id: str
    ) -> Storyline:
        """
        Build storyline from clustered events.

        Args:
            events: Events in this storyline
            cluster_idx: Cluster index
            batch_id: Batch identifier

        Returns:
            Storyline object
        """
        storyline_id = create_storyline_id(batch_id, cluster_idx)

        # Extract primary entities (most frequent across events)
        entity_counts = defaultdict(int)
        for event in events:
            for arg in event.arguments:
                entity_counts[arg.entity.text.lower()] += 1

        # Top 5 entities
        primary_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        primary_entities = [entity for entity, _ in primary_entities]

        # Determine primary domain
        domain_counts = defaultdict(int)
        for event in events:
            if event.domain:
                domain_counts[event.domain] += 1

        primary_domain = max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else None

        # Extract temporal span
        temporal_refs = [e.temporal_reference for e in events if e.temporal_reference]
        temporal_span = None
        if temporal_refs:
            try:
                dates = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in temporal_refs]
                min_date = min(dates).isoformat()
                max_date = max(dates).isoformat()
                temporal_span = (min_date, max_date)
            except Exception:
                pass

        # Build storyline
        storyline = Storyline(
            storyline_id=storyline_id,
            event_ids=[e.event_id for e in events],
            primary_entities=primary_entities,
            domain=primary_domain,
            temporal_span=temporal_span,
            storyline_summary=self._generate_storyline_summary(events, primary_entities, primary_domain)
        )

        return storyline

    def _generate_storyline_summary(
        self,
        events: List[Event],
        primary_entities: List[str],
        primary_domain: Optional[str]
    ) -> str:
        """Generate brief storyline summary."""
        entities_str = ", ".join(primary_entities[:3])
        domain_str = primary_domain.replace("_", " ").title() if primary_domain else "General"
        return f"{domain_str} storyline involving {entities_str} ({len(events)} events)"

    def _assign_storyline_ids(self, events: List[Event], storylines: List[Storyline]):
        """Assign storyline IDs to events."""
        event_to_storyline = {}
        for storyline in storylines:
            for event_id in storyline.event_ids:
                event_to_storyline[event_id] = storyline.storyline_id

        for event in events:
            if event.event_id in event_to_storyline:
                event.storyline_id = event_to_storyline[event.event_id]

    def _detect_causal_chains(
        self,
        events: List[Event],
        linkages: List[EventLinkage],
        storylines: List[Storyline]
    ):
        """
        Detect causal relationships and chains within storylines.

        Updates event metadata with causality information.
        """
        if not self.settings.enable_causality_detection:
            return

        # Build causality indicators pattern
        indicators = "|".join(self.settings.causality_indicators)

        # Check each event for causality indicators
        for event in events:
            if event.metadata.causality:
                # Already has causality from LLM
                continue

            # Check if metadata contains causality indicators
            source_text = event.metadata.source_sentence or ""
            for indicator in self.settings.causality_indicators:
                if indicator in source_text.lower():
                    event.metadata.causality = f"Contains causal indicator: {indicator}"
                    break


# =============================================================================
# Singleton Instance
# =============================================================================

_linker_instance: Optional[EventLinker] = None


def get_event_linker() -> EventLinker:
    """
    Get singleton event linker instance.

    Returns:
        EventLinker instance
    """
    global _linker_instance
    if _linker_instance is None:
        _linker_instance = EventLinker()
    return _linker_instance


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Test event linker
    try:
        linker = get_event_linker()
        print("Event linker initialized successfully")
        print(f"Embedding model: {linker.settings.embedding_model}")
        print(f"Domain boundaries enforced: {linker.settings.enforce_domain_boundaries}")

    except Exception as e:
        print(f"Failed to test event linker: {e}")
        sys.exit(1)
