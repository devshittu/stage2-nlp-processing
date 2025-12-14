"""
llm_prompts.py

Domain-aware LLM prompts for event extraction in Stage 2 NLP Processing Service.

This module provides:
1. Domain-specific system prompts optimized for storyline distinction
2. Event type definitions based on ACE 2005 extended taxonomy
3. Argument role definitions for event participants
4. Few-shot examples for each domain
5. Prompt building and LLM output parsing functions

Key Features:
- Compact prompts optimized for token efficiency
- Multi-dimensional event extraction (type, trigger, arguments, sentiment, causality)
- Domain classification to prevent storyline conflation (e.g., Trump+Israel vs Trump+Qatar)
- Entity-role-context awareness for disambiguation
- Structured JSON output format compatible with vLLM and Mistral models

Storyline Distinction Approach:
- Each event is tagged with a domain (geopolitical_conflict, diplomatic_relations, etc.)
- Events are enriched with entity-role-context to prevent false linkage
- Example: "Trump meeting with Netanyahu" (diplomatic_relations, Israel context)
           vs "Trump negotiating with Qatar" (diplomatic_relations, Qatar context)
- Downstream event linking uses domain + entity overlap + temporal proximity
- Causality chains are extracted to connect events within the same storyline
"""

import json
import re
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.schemas.data_models import (
    Event,
    EventTrigger,
    EventArgument,
    EventMetadata,
    Entity,
    create_event_id
)
from src.utils.config_manager import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are an expert event extraction system. Extract ONLY MAIN, NEWSWORTHY, SEMANTICALLY SIGNIFICANT events from news articles for storyline construction.

CORE PRINCIPLE: EXTRACT REAL-WORLD HAPPENINGS, NOT REPORTING ACTIONS
Extract what HAPPENED in the real world, not the meta-actions of reporting it.

TASK:
1. Identify MAIN NEWSWORTHY EVENTS - significant real-world happenings that drive the narrative
2. For each event, extract:
   - Event type (from predefined taxonomy)
   - Trigger word/phrase (word that indicates the actual event)
   - Arguments (participants with roles: agent, patient, time, place, etc.)
   - Domain classification (geopolitical_conflict, diplomatic_relations, etc.)
   - Sentiment (positive, negative, neutral, mixed)
   - Causality (explain what caused this event or what it led to)

CRITICAL REQUIREMENTS - QUALITY OVER QUANTITY:
⚠️ EXTRACT ONLY MAIN NEWSWORTHY EVENTS - typically 1-4 per document, rarely more than 8
⚠️ Ask yourself: "Is this a significant real-world happening worthy of news coverage?"
⚠️ DO NOT extract every verb/action - focus on what actually HAPPENED in the real world

WHAT TO EXTRACT (Real-World Events):
✓ Physical actions: meetings, attacks, agreements, elections, appointments, deaths, disasters
✓ Policy changes: laws signed, regulations implemented, treaties ratified
✓ Transactions: acquisitions, donations, investments, sales
✓ Legal actions: arrests, trials, convictions, court rulings
✓ Personnel changes: hirings, resignations, elections to positions

WHAT TO IGNORE (Meta-Reporting Actions):
✗ Reporting verbs: "said", "announced", "reported", "stated", "claimed", "revealed", "disclosed"
   → Extract the CONTENT of what was said/announced, not the act of saying
   → Example: "Biden announced new climate policy" → Extract the policy announcement, not "announced"
✗ Communication verbs: "told", "mentioned", "indicated", "noted", "added", "commented"
✗ Attribution verbs: "according to", "sources say", "officials claim"
✗ State-of-being: "is", "was", "has been", "remains", "continues to be"
✗ Descriptive statements: "is controversial", "seems likely", "appears to be"
✗ Background information: "has a history of", "previously worked at"

CONCRETE EXAMPLES:
✓ EXTRACT: "Biden met with Netanyahu in Washington" → contact_meet (real meeting happened)
✗ IGNORE: "Officials said Biden met Netanyahu" → Don't extract "said", extract the meeting
✗ IGNORE: "Biden announced he would meet Netanyahu" → Don't extract "announced", wait for actual meeting

✓ EXTRACT: "Israel launched airstrikes on Gaza" → conflict_attack (real attack happened)
✗ IGNORE: "Reports claim Israel launched strikes" → Don't extract "claim", extract the attack if confirmed

✓ EXTRACT: "The president signed the bill into law" → policy_implement (real action)
✗ IGNORE: "The bill is controversial" → Descriptive, not an event
✗ IGNORE: "Lawmakers said the bill would pass" → Don't extract "said", extract signing when it happens

✓ EXTRACT: "Microsoft acquired OpenAI for $10B" → transaction_transfer_ownership (real acquisition)
✗ IGNORE: "Microsoft announced plans to acquire OpenAI" → "announced plans" is not the event, extract when acquisition happens

QUALITY GUIDELINES:
- Better to extract too few events than too many - focus on what truly matters
- Each event should be independently newsworthy
- Avoid granular sub-actions (e.g., if there's a meeting, don't also extract "arrived", "discussed", "left")
- Prioritize events that advance the storyline
- A typical news article has 1-4 main events, not 10+

OUTPUT FORMAT:
Return JSON object with "events" array. Each event must include:
{
  "events": [
    {
      "event_type": "policy_announce",
      "trigger": {"text": "announced", "start_char": 45, "end_char": 54},
      "arguments": [
        {"role": "agent", "text": "President Biden", "start_char": 0, "end_char": 15, "type": "PER"},
        {"role": "patient", "text": "new climate policy", "start_char": 55, "end_char": 73, "type": "MISC"},
        {"role": "time", "text": "yesterday", "start_char": 74, "end_char": 83, "type": "DATE"},
        {"role": "place", "text": "White House", "start_char": 91, "end_char": 102, "type": "LOC"}
      ],
      "domain": "environmental_climate",
      "sentiment": "positive",
      "causality": "Response to recent climate summit recommendations"
    }
  ]
}

IMPORTANT:
- character positions (start_char/end_char) must be exact offsets in the input text
- end_char is exclusive (points to character after the span)
- If multiple events share entities, extract separately with full context
- Domain should reflect the PRIMARY nature of the event (not secondary aspects)
"""


# =============================================================================
# DOMAIN-SPECIFIC PROMPTS
# =============================================================================

DOMAIN_SPECIFIC_PROMPTS: Dict[str, str] = {
    "geopolitical_conflict": """
DOMAIN: Geopolitical Conflict
Focus on: Military actions, armed conflicts, wars, airstrikes, invasions, territorial disputes, defense operations.
Key event types: conflict_attack, conflict_demonstrate, life_die, life_injure, movement_transport (military).
Example entities: Military units, countries, weapons, casualties, conflict zones.
Context markers: Pay attention to location context, military roles, and casualty details.
""",

    "diplomatic_relations": """
DOMAIN: Diplomatic Relations
Focus on: International meetings, treaties, negotiations, diplomatic visits, international agreements, summits.
Key event types: contact_meet, agreement_sign, agreement_negotiate, contact_phone_write.
Example entities: World leaders, diplomats, international organizations, countries, diplomatic venues.
Context markers: Distinguish meetings by participants AND topics (e.g., US-Israel talks vs US-Qatar talks).
""",

    "economic_policy": """
DOMAIN: Economic Policy
Focus on: Trade agreements, tariffs, economic sanctions, monetary policy, international trade, economic partnerships.
Key event types: transaction_transfer_money, agreement_sign, policy_announce, policy_implement.
Example entities: Central banks, trade organizations, currencies, economic zones, companies.
Context markers: Track economic actors, amounts, affected countries/sectors, and policy rationale.
""",

    "domestic_policy": """
DOMAIN: Domestic Policy
Focus on: Internal government policy changes, legislation, executive orders, regulatory changes, domestic reforms.
Key event types: policy_announce, policy_implement, policy_change, personnel_elect.
Example entities: Government agencies, politicians, legislative bodies, affected populations, policy areas.
Context markers: Distinguish policies by country/region, policy domain, and implementing authority.
""",

    "elections_politics": """
DOMAIN: Elections & Politics
Focus on: Electoral events, campaigns, political appointments, resignations, party activities, voting.
Key event types: personnel_elect, personnel_start_position, personnel_end_position, conflict_demonstrate.
Example entities: Politicians, political parties, voters, electoral districts, government positions.
Context markers: Track candidate names, positions contested, political affiliations, election dates.
""",

    "technology_innovation": """
DOMAIN: Technology & Innovation
Focus on: Tech product launches, AI developments, cybersecurity events, tech regulations, innovation breakthroughs.
Key event types: policy_announce, transaction_transfer_ownership, agreement_sign, policy_implement.
Example entities: Tech companies, products, technologies, regulatory bodies, markets.
Context markers: Distinguish by technology type, companies involved, regulatory jurisdictions.
""",

    "social_movements": """
DOMAIN: Social Movements
Focus on: Protests, demonstrations, civil rights events, social activism, public mobilization, strikes.
Key event types: conflict_demonstrate, contact_meet, personnel_elect, policy_change.
Example entities: Activist groups, protest locations, movement leaders, affected communities, demands.
Context markers: Track movement goals, participant groups, locations, and governmental responses.
""",

    "environmental_climate": """
DOMAIN: Environmental & Climate
Focus on: Climate policy, environmental disasters, conservation efforts, emissions agreements, extreme weather.
Key event types: policy_announce, agreement_sign, conflict_demonstrate, life_die (disasters).
Example entities: Environmental agencies, affected regions, climate metrics, international bodies, ecosystems.
Context markers: Note environmental impact, affected areas, policy targets, scientific findings.
""",

    "health_pandemic": """
DOMAIN: Health & Pandemic
Focus on: Disease outbreaks, public health policy, vaccination campaigns, healthcare crises, medical breakthroughs.
Key event types: policy_announce, policy_implement, life_die, life_injure, agreement_sign.
Example entities: Health organizations, diseases, medications, healthcare facilities, affected populations.
Context markers: Track disease types, affected regions, health measures, casualty numbers.
""",

    "legal_judicial": """
DOMAIN: Legal & Judicial
Focus on: Court rulings, trials, arrests, legal proceedings, judicial appointments, landmark cases.
Key event types: justice_arrest, justice_trial, justice_convict, personnel_start_position.
Example entities: Courts, judges, defendants, lawyers, legal charges, verdicts.
Context markers: Distinguish by jurisdiction, case type, legal parties, charges/verdicts.
""",

    "corporate_business": """
DOMAIN: Corporate & Business
Focus on: Mergers, acquisitions, business deals, corporate leadership changes, bankruptcies, earnings.
Key event types: transaction_transfer_ownership, transaction_transfer_money, personnel_start_position, personnel_end_position.
Example entities: Companies, executives, investors, markets, products, financial amounts.
Context markers: Track company names, deal values, business sectors, geographic markets.
""",

    "cultural_entertainment": """
DOMAIN: Cultural & Entertainment
Focus on: Cultural events, entertainment releases, awards, celebrity news, artistic performances, festivals.
Key event types: contact_meet, personnel_start_position, personnel_end_position, policy_announce.
Example entities: Artists, cultural venues, productions, awards, cultural organizations, audiences.
Context markers: Distinguish by event type, participants, cultural context, geographic location.
"""
}


# =============================================================================
# EVENT TYPE DEFINITIONS (ACE 2005 + Extended)
# =============================================================================

EVENT_TYPE_DEFINITIONS: Dict[str, str] = {
    # Conflict Events
    "conflict_attack": "Physical attack, military strike, airstrike, bombing, assault, combat action",
    "conflict_demonstrate": "Protest, demonstration, rally, civil unrest, strike, public gathering",

    # Contact Events
    "contact_meet": "Meeting, summit, conference, discussion, diplomatic encounter, face-to-face interaction",
    "contact_phone_write": "Phone call, video call, written communication, correspondence, message exchange",

    # Justice Events
    "justice_arrest": "Arrest, detention, custody, law enforcement apprehension",
    "justice_trial": "Trial, hearing, legal proceeding, court case, judicial process",
    "justice_convict": "Conviction, sentencing, verdict, judicial ruling, legal judgment",

    # Life Events
    "life_die": "Death, killing, casualty, fatality, loss of life",
    "life_injure": "Injury, wounding, harm, medical trauma, physical damage to persons",

    # Movement Events
    "movement_transport": "Transportation, travel, deployment, evacuation, relocation, movement of people/goods",

    # Personnel Events
    "personnel_elect": "Election, selection, voting, democratic choice, appointment via election",
    "personnel_start_position": "Hiring, appointment, assumption of position, job start, inauguration",
    "personnel_end_position": "Resignation, firing, retirement, removal from position, job termination",

    # Transaction Events
    "transaction_transfer_money": "Payment, financial transfer, funding, investment, monetary transaction",
    "transaction_transfer_ownership": "Sale, acquisition, purchase, merger, ownership change, asset transfer",

    # Policy Events (Extended)
    "policy_announce": "Policy announcement, declaration of new policy, official statement of intent",
    "policy_implement": "Policy implementation, execution of policy, putting policy into effect",
    "policy_change": "Policy modification, revision of existing policy, regulatory change",

    # Agreement Events (Extended)
    "agreement_sign": "Treaty signing, contract signing, formal agreement, deal finalization",
    "agreement_negotiate": "Negotiation, bargaining, diplomatic discussion, deal-making process",
}


# =============================================================================
# ARGUMENT ROLE DEFINITIONS
# =============================================================================

ARGUMENT_ROLE_DEFINITIONS: Dict[str, str] = {
    "agent": "The entity performing the action (who did it). Usually a person or organization.",
    "patient": "The entity affected by the action (who/what it happened to). Can be person, object, or concept.",
    "time": "When the event occurred. Can be absolute (dates) or relative (yesterday, recently).",
    "place": "Where the event occurred. Location, venue, geographic entity, or facility.",
    "instrument": "How the action was performed or with what tool/method. Weapons, tools, mechanisms.",
    "beneficiary": "Who/what benefits from the event. The recipient or intended beneficiary.",
    "purpose": "Why the event occurred or what goal it serves. Motivation, intention, or objective.",
}


# =============================================================================
# FEW-SHOT EXAMPLES (Compact, Domain-Specific)
# =============================================================================

FEW_SHOT_EXAMPLES: Dict[str, List[Dict[str, Any]]] = {
    "geopolitical_conflict": [
        {
            "text": "Israeli forces launched airstrikes on Gaza targets early Tuesday, killing at least 15 people.",
            "output": {
                "events": [
                    {
                        "event_type": "conflict_attack",
                        "trigger": {"text": "airstrikes", "start_char": 23, "end_char": 33},
                        "arguments": [
                            {"role": "agent", "text": "Israeli forces", "start_char": 0, "end_char": 14, "type": "ORG"},
                            {"role": "patient", "text": "Gaza targets", "start_char": 37, "end_char": 49, "type": "LOC"},
                            {"role": "time", "text": "early Tuesday", "start_char": 50, "end_char": 63, "type": "DATE"}
                        ],
                        "domain": "geopolitical_conflict",
                        "sentiment": "negative",
                        "causality": "Part of ongoing regional conflict"
                    },
                    {
                        "event_type": "life_die",
                        "trigger": {"text": "killing", "start_char": 65, "end_char": 72},
                        "arguments": [
                            {"role": "agent", "text": "Israeli forces", "start_char": 0, "end_char": 14, "type": "ORG"},
                            {"role": "patient", "text": "at least 15 people", "start_char": 73, "end_char": 91, "type": "PER"},
                            {"role": "place", "text": "Gaza", "start_char": 37, "end_char": 41, "type": "GPE"}
                        ],
                        "domain": "geopolitical_conflict",
                        "sentiment": "negative",
                        "causality": "Direct result of airstrikes on Gaza targets"
                    }
                ]
            }
        }
    ],

    "diplomatic_relations": [
        {
            "text": "President Biden met with Israeli PM Netanyahu in Washington yesterday to discuss regional security cooperation.",
            "output": {
                "events": [
                    {
                        "event_type": "contact_meet",
                        "trigger": {"text": "met", "start_char": 16, "end_char": 19},
                        "arguments": [
                            {"role": "agent", "text": "President Biden", "start_char": 0, "end_char": 15, "type": "PER"},
                            {"role": "patient", "text": "Israeli PM Netanyahu", "start_char": 25, "end_char": 45, "type": "PER"},
                            {"role": "place", "text": "Washington", "start_char": 49, "end_char": 59, "type": "GPE"},
                            {"role": "time", "text": "yesterday", "start_char": 60, "end_char": 69, "type": "DATE"},
                            {"role": "purpose", "text": "discuss regional security cooperation", "start_char": 73, "end_char": 110, "type": "MISC"}
                        ],
                        "domain": "diplomatic_relations",
                        "sentiment": "neutral",
                        "causality": "Scheduled diplomatic engagement for bilateral security discussions"
                    }
                ]
            }
        }
    ],

    "economic_policy": [
        {
            "text": "The EU announced new tariffs on Chinese electric vehicles, effective next month.",
            "output": {
                "events": [
                    {
                        "event_type": "policy_announce",
                        "trigger": {"text": "announced", "start_char": 7, "end_char": 16},
                        "arguments": [
                            {"role": "agent", "text": "The EU", "start_char": 0, "end_char": 6, "type": "ORG"},
                            {"role": "patient", "text": "new tariffs on Chinese electric vehicles", "start_char": 17, "end_char": 57, "type": "MISC"},
                            {"role": "time", "text": "next month", "start_char": 69, "end_char": 79, "type": "DATE"}
                        ],
                        "domain": "economic_policy",
                        "sentiment": "neutral",
                        "causality": "Response to concerns about fair trade and market competition"
                    }
                ]
            }
        }
    ],

    "domestic_policy": [
        {
            "text": "The UK government implemented stricter immigration controls following last year's legislative changes.",
            "output": {
                "events": [
                    {
                        "event_type": "policy_implement",
                        "trigger": {"text": "implemented", "start_char": 18, "end_char": 29},
                        "arguments": [
                            {"role": "agent", "text": "The UK government", "start_char": 0, "end_char": 17, "type": "ORG"},
                            {"role": "patient", "text": "stricter immigration controls", "start_char": 30, "end_char": 59, "type": "MISC"},
                            {"role": "time", "text": "following last year's legislative changes", "start_char": 60, "end_char": 101, "type": "DATE"}
                        ],
                        "domain": "domestic_policy",
                        "sentiment": "neutral",
                        "causality": "Implementation of last year's legislative changes on immigration"
                    }
                ]
            }
        }
    ],

    "elections_politics": [
        {
            "text": "Senator Martinez was elected as majority leader by the Senate Democratic caucus on Tuesday.",
            "output": {
                "events": [
                    {
                        "event_type": "personnel_elect",
                        "trigger": {"text": "elected", "start_char": 21, "end_char": 28},
                        "arguments": [
                            {"role": "patient", "text": "Senator Martinez", "start_char": 0, "end_char": 16, "type": "PER"},
                            {"role": "agent", "text": "Senate Democratic caucus", "start_char": 55, "end_char": 79, "type": "ORG"},
                            {"role": "purpose", "text": "majority leader", "start_char": 32, "end_char": 47, "type": "MISC"},
                            {"role": "time", "text": "Tuesday", "start_char": 83, "end_char": 90, "type": "DATE"}
                        ],
                        "domain": "elections_politics",
                        "sentiment": "neutral",
                        "causality": "Result of internal party leadership election"
                    }
                ]
            }
        }
    ],

    "technology_innovation": [
        {
            "text": "The European Commission announced sweeping AI regulations requiring transparency from tech companies.",
            "output": {
                "events": [
                    {
                        "event_type": "policy_announce",
                        "trigger": {"text": "announced", "start_char": 24, "end_char": 33},
                        "arguments": [
                            {"role": "agent", "text": "The European Commission", "start_char": 0, "end_char": 23, "type": "ORG"},
                            {"role": "patient", "text": "sweeping AI regulations", "start_char": 34, "end_char": 57, "type": "MISC"},
                            {"role": "beneficiary", "text": "tech companies", "start_char": 86, "end_char": 100, "type": "ORG"},
                            {"role": "purpose", "text": "requiring transparency", "start_char": 58, "end_char": 80, "type": "MISC"}
                        ],
                        "domain": "technology_innovation",
                        "sentiment": "neutral",
                        "causality": "Response to rapid AI development and need for governance"
                    }
                ]
            }
        }
    ],

    "social_movements": [
        {
            "text": "Thousands of climate activists protested outside the UN headquarters demanding immediate climate action.",
            "output": {
                "events": [
                    {
                        "event_type": "conflict_demonstrate",
                        "trigger": {"text": "protested", "start_char": 31, "end_char": 40},
                        "arguments": [
                            {"role": "agent", "text": "Thousands of climate activists", "start_char": 0, "end_char": 30, "type": "PER"},
                            {"role": "place", "text": "UN headquarters", "start_char": 53, "end_char": 68, "type": "LOC"},
                            {"role": "purpose", "text": "demanding immediate climate action", "start_char": 69, "end_char": 103, "type": "MISC"}
                        ],
                        "domain": "social_movements",
                        "sentiment": "neutral",
                        "causality": "Response to perceived insufficient climate policy action"
                    }
                ]
            }
        }
    ],

    "environmental_climate": [
        {
            "text": "World leaders signed a landmark climate agreement at COP29, committing to net-zero emissions by 2050.",
            "output": {
                "events": [
                    {
                        "event_type": "agreement_sign",
                        "trigger": {"text": "signed", "start_char": 14, "end_char": 20},
                        "arguments": [
                            {"role": "agent", "text": "World leaders", "start_char": 0, "end_char": 13, "type": "PER"},
                            {"role": "patient", "text": "landmark climate agreement", "start_char": 23, "end_char": 49, "type": "MISC"},
                            {"role": "place", "text": "COP29", "start_char": 53, "end_char": 58, "type": "EVENT"},
                            {"role": "purpose", "text": "committing to net-zero emissions by 2050", "start_char": 60, "end_char": 100, "type": "MISC"}
                        ],
                        "domain": "environmental_climate",
                        "sentiment": "positive",
                        "causality": "Response to global climate crisis and international pressure"
                    }
                ]
            }
        }
    ],

    "health_pandemic": [
        {
            "text": "WHO declared a global health emergency after the novel virus spread to 30 countries within two weeks.",
            "output": {
                "events": [
                    {
                        "event_type": "policy_announce",
                        "trigger": {"text": "declared", "start_char": 4, "end_char": 12},
                        "arguments": [
                            {"role": "agent", "text": "WHO", "start_char": 0, "end_char": 3, "type": "ORG"},
                            {"role": "patient", "text": "global health emergency", "start_char": 15, "end_char": 38, "type": "MISC"},
                            {"role": "time", "text": "within two weeks", "start_char": 84, "end_char": 100, "type": "DATE"}
                        ],
                        "domain": "health_pandemic",
                        "sentiment": "negative",
                        "causality": "Response to rapid international spread of novel virus to 30 countries"
                    }
                ]
            }
        }
    ],

    "legal_judicial": [
        {
            "text": "The Supreme Court convicted the former minister of corruption charges, sentencing him to 10 years in prison.",
            "output": {
                "events": [
                    {
                        "event_type": "justice_convict",
                        "trigger": {"text": "convicted", "start_char": 18, "end_char": 27},
                        "arguments": [
                            {"role": "agent", "text": "The Supreme Court", "start_char": 0, "end_char": 17, "type": "ORG"},
                            {"role": "patient", "text": "the former minister", "start_char": 28, "end_char": 47, "type": "PER"},
                            {"role": "instrument", "text": "corruption charges", "start_char": 51, "end_char": 69, "type": "MISC"},
                            {"role": "purpose", "text": "10 years in prison", "start_char": 89, "end_char": 107, "type": "MISC"}
                        ],
                        "domain": "legal_judicial",
                        "sentiment": "neutral",
                        "causality": "Result of judicial proceedings on corruption allegations"
                    }
                ]
            }
        }
    ],

    "corporate_business": [
        {
            "text": "Tech giant Microsoft acquired AI startup OpenAI for $10 billion, announced CEO Nadella yesterday.",
            "output": {
                "events": [
                    {
                        "event_type": "transaction_transfer_ownership",
                        "trigger": {"text": "acquired", "start_char": 21, "end_char": 29},
                        "arguments": [
                            {"role": "agent", "text": "Microsoft", "start_char": 11, "end_char": 20, "type": "ORG"},
                            {"role": "patient", "text": "AI startup OpenAI", "start_char": 30, "end_char": 47, "type": "ORG"},
                            {"role": "instrument", "text": "$10 billion", "start_char": 52, "end_char": 63, "type": "MONEY"},
                            {"role": "time", "text": "yesterday", "start_char": 87, "end_char": 96, "type": "DATE"}
                        ],
                        "domain": "corporate_business",
                        "sentiment": "positive",
                        "causality": "Strategic acquisition to strengthen AI capabilities"
                    }
                ]
            }
        }
    ],

    "cultural_entertainment": [
        {
            "text": "Director Jane Smith won the Best Director award at the Cannes Film Festival for her documentary.",
            "output": {
                "events": [
                    {
                        "event_type": "personnel_elect",
                        "trigger": {"text": "won", "start_char": 20, "end_char": 23},
                        "arguments": [
                            {"role": "patient", "text": "Director Jane Smith", "start_char": 0, "end_char": 19, "type": "PER"},
                            {"role": "purpose", "text": "Best Director award", "start_char": 28, "end_char": 47, "type": "MISC"},
                            {"role": "place", "text": "Cannes Film Festival", "start_char": 55, "end_char": 75, "type": "EVENT"},
                            {"role": "instrument", "text": "her documentary", "start_char": 80, "end_char": 95, "type": "MISC"}
                        ],
                        "domain": "cultural_entertainment",
                        "sentiment": "positive",
                        "causality": "Recognition of artistic achievement at international film festival"
                    }
                ]
            }
        }
    ]
}


# =============================================================================
# PROMPT BUILDING FUNCTION
# =============================================================================

def build_prompt(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    include_examples: bool = True
) -> str:
    """
    Build domain-aware prompt for event extraction.

    Args:
        text: The input text to extract events from
        context: Optional context fields (title, author, date, etc.) from Stage 1
        domain: Optional domain hint for domain-specific instructions
        include_examples: Whether to include few-shot examples (increases token count)

    Returns:
        Formatted prompt string ready for LLM inference

    Example:
        >>> context = {"cleaned_title": "Biden Meets Netanyahu", "cleaned_publication_date": "2024-01-15"}
        >>> prompt = build_prompt(text, context=context, domain="diplomatic_relations")
    """
    settings = get_settings()

    # Start with base system prompt
    prompt_parts = [SYSTEM_PROMPT_TEMPLATE]

    # Add domain-specific instructions if domain is provided
    if domain and domain in DOMAIN_SPECIFIC_PROMPTS:
        prompt_parts.append(DOMAIN_SPECIFIC_PROMPTS[domain])
    else:
        # If no domain specified, provide general guidance on all domains
        prompt_parts.append("\nEXTRACT FROM ANY OF THESE DOMAINS:\n")
        prompt_parts.append(", ".join(settings.event_llm_service.domains))

    # Add event type definitions (compact format)
    prompt_parts.append("\n\nEVENT TYPES:")
    for event_type, description in EVENT_TYPE_DEFINITIONS.items():
        prompt_parts.append(f"- {event_type}: {description}")

    # Add argument role definitions (compact format)
    prompt_parts.append("\n\nARGUMENT ROLES:")
    for role, description in ARGUMENT_ROLE_DEFINITIONS.items():
        prompt_parts.append(f"- {role}: {description}")

    # Add few-shot examples if requested and domain is specific
    if include_examples and domain and domain in FEW_SHOT_EXAMPLES:
        prompt_parts.append(f"\n\nEXAMPLES FOR {domain.upper()}:")
        for idx, example in enumerate(FEW_SHOT_EXAMPLES[domain][:2], 1):  # Max 2 examples for token efficiency
            prompt_parts.append(f"\nExample {idx}:")
            prompt_parts.append(f"INPUT: {example['text']}")
            prompt_parts.append(f"OUTPUT: {json.dumps(example['output'], indent=2)}")

    # Add context information if available
    if context:
        prompt_parts.append("\n\nCONTEXT INFORMATION:")
        if "cleaned_title" in context and context["cleaned_title"]:
            prompt_parts.append(f"Title: {context['cleaned_title']}")
        if "cleaned_publication_date" in context and context["cleaned_publication_date"]:
            prompt_parts.append(f"Publication Date: {context['cleaned_publication_date']}")
        if "cleaned_author" in context and context["cleaned_author"]:
            prompt_parts.append(f"Author: {context['cleaned_author']}")
        if "cleaned_categories" in context and context["cleaned_categories"]:
            prompt_parts.append(f"Categories: {', '.join(context['cleaned_categories'])}")

    # Add the actual text to process
    prompt_parts.append("\n\n=== TEXT TO ANALYZE ===")
    prompt_parts.append(text)
    prompt_parts.append("\n=== END OF TEXT ===")

    # Final instruction
    prompt_parts.append("\nExtract ONLY the MAIN NEWSWORTHY events from the text above (typically 1-4 events). Return ONLY valid JSON, no additional text.")

    return "\n".join(prompt_parts)


# =============================================================================
# LLM OUTPUT PARSING FUNCTION
# =============================================================================

def parse_llm_output(
    output: str,
    document_id: str,
    original_text: str,
    default_domain: Optional[str] = None
) -> List[Event]:
    """
    Parse LLM output and convert to Event objects.

    Handles:
    - JSON extraction from LLM output (removes markdown, extra text)
    - Validation of event structure
    - Creation of Event, EventTrigger, EventArgument objects
    - Fallback values for missing fields
    - Character position validation

    Args:
        output: Raw LLM output (may contain JSON in markdown or mixed with text)
        document_id: Document identifier for event ID generation
        original_text: Original text for character position validation
        default_domain: Default domain if not specified in event

    Returns:
        List of Event objects

    Raises:
        ValueError: If output cannot be parsed or is invalid
    """
    try:
        # Extract JSON from output (handle markdown code blocks)
        json_str = _extract_json_from_output(output)

        # Parse JSON
        parsed = json.loads(json_str)

        # Validate structure
        if "events" not in parsed:
            logger.warning("No 'events' key in LLM output, attempting to treat entire output as events array")
            if isinstance(parsed, list):
                parsed = {"events": parsed}
            else:
                raise ValueError("Invalid output structure: missing 'events' key")

        events_data = parsed["events"]
        if not isinstance(events_data, list):
            raise ValueError("'events' must be a list")

        # Debug logging
        logger.debug(f"Found {len(events_data)} events in LLM output")
        logger.debug(f"Events data types: {[type(e).__name__ for e in events_data[:3]]}")

        # Convert to Event objects
        events = []
        for idx, event_dict in enumerate(events_data):
            try:
                # Validate event_dict is a dictionary
                if not isinstance(event_dict, dict):
                    logger.warning(f"Event {idx} is not a dictionary, got {type(event_dict).__name__}: {event_dict}. Skipping.")
                    continue

                logger.debug(f"Parsing event {idx}, original_text type: {type(original_text).__name__}")
                event = _parse_single_event(
                    event_dict=event_dict,
                    document_id=document_id,
                    event_index=idx,
                    original_text=original_text,
                    default_domain=default_domain
                )
                events.append(event)
            except Exception as e:
                import traceback
                logger.warning(f"Failed to parse event {idx}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                continue

        logger.info(f"Successfully parsed {len(events)} events from LLM output")
        return events

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM output: {e}")
        logger.debug(f"Problematic output: {output[:500]}...")
        raise ValueError(f"Invalid JSON in LLM output: {e}")

    except Exception as e:
        logger.error(f"Failed to parse LLM output: {e}")
        raise ValueError(f"Failed to parse LLM output: {e}")


def _extract_json_from_output(output: str) -> str:
    """
    Extract JSON from LLM output, handling markdown code blocks and extra text.

    Args:
        output: Raw LLM output

    Returns:
        Clean JSON string
    """
    # Remove markdown code blocks
    output = re.sub(r'```json\s*', '', output)
    output = re.sub(r'```\s*', '', output)

    # Try to find JSON object or array
    # Look for opening brace/bracket
    json_start = -1
    for i, char in enumerate(output):
        if char in ['{', '[']:
            json_start = i
            break

    if json_start == -1:
        raise ValueError("No JSON object or array found in output")

    # Find matching closing brace/bracket
    output = output[json_start:]

    # Try to parse incrementally to find valid JSON
    for i in range(len(output), 0, -1):
        try:
            test_str = output[:i]
            json.loads(test_str)
            return test_str
        except json.JSONDecodeError:
            continue

    # If nothing works, return as-is and let the caller handle the error
    return output


def _parse_single_event(
    event_dict: Dict[str, Any],
    document_id: str,
    event_index: int,
    original_text: str,
    default_domain: Optional[str] = None
) -> Event:
    """
    Parse a single event dictionary into an Event object.

    Args:
        event_dict: Event data from LLM
        document_id: Document identifier
        event_index: Event index for ID generation
        original_text: Original text for validation
        default_domain: Default domain if not in event

    Returns:
        Event object

    Raises:
        ValueError: If event data is invalid
    """
    # Validate required fields
    if "event_type" not in event_dict:
        raise ValueError("Missing required field: event_type")
    if "trigger" not in event_dict:
        raise ValueError("Missing required field: trigger")

    # Parse trigger
    trigger_data = event_dict["trigger"]
    trigger = EventTrigger(
        text=trigger_data.get("text", ""),
        start_char=trigger_data.get("start_char", 0),
        end_char=trigger_data.get("end_char", 0),
        lemma=trigger_data.get("lemma")
    )

    # Validate trigger position
    if not _validate_span(trigger.start_char, trigger.end_char, len(original_text)):
        logger.warning(f"Invalid trigger span: [{trigger.start_char}, {trigger.end_char})")

    # Parse arguments
    arguments = []
    for arg_data in event_dict.get("arguments", []):
        try:
            entity = Entity(
                text=arg_data.get("text", ""),
                type=arg_data.get("type", "MISC"),
                start_char=arg_data.get("start_char", 0),
                end_char=arg_data.get("end_char", 0),
                confidence=arg_data.get("confidence", 1.0)
            )

            argument = EventArgument(
                argument_role=arg_data.get("role", "unknown"),
                entity=entity,
                confidence=arg_data.get("confidence", 1.0)
            )

            # Validate argument span
            if not _validate_span(entity.start_char, entity.end_char, len(original_text)):
                logger.warning(f"Invalid argument span: [{entity.start_char}, {entity.end_char})")

            arguments.append(argument)
        except Exception as e:
            logger.warning(f"Failed to parse argument: {e}. Skipping.")
            continue

    # Parse metadata
    metadata = EventMetadata(
        sentiment=event_dict.get("sentiment", "neutral"),
        causality=event_dict.get("causality"),
        confidence=event_dict.get("confidence", 1.0),
        source_sentence=event_dict.get("source_sentence")
    )

    # Create Event object
    event = Event(
        event_id=create_event_id(document_id, event_index),
        event_type=event_dict["event_type"],
        trigger=trigger,
        arguments=arguments,
        metadata=metadata,
        domain=event_dict.get("domain", default_domain),
        domain_confidence=event_dict.get("domain_confidence"),
        temporal_reference=event_dict.get("temporal_reference")
    )

    return event


def _validate_span(start_char: int, end_char: int, text_length: int) -> bool:
    """
    Validate character span positions.

    Args:
        start_char: Start position
        end_char: End position (exclusive)
        text_length: Total text length

    Returns:
        True if valid, False otherwise
    """
    if start_char < 0 or end_char < 0:
        return False
    if start_char >= end_char:
        return False
    if end_char > text_length:
        return False
    return True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_domain_prompt(domain: str) -> Optional[str]:
    """
    Get domain-specific prompt instructions.

    Args:
        domain: Domain identifier

    Returns:
        Domain-specific prompt or None if domain not found
    """
    return DOMAIN_SPECIFIC_PROMPTS.get(domain)


def list_supported_domains() -> List[str]:
    """
    Get list of supported domains.

    Returns:
        List of domain identifiers
    """
    return list(DOMAIN_SPECIFIC_PROMPTS.keys())


def list_supported_event_types() -> List[str]:
    """
    Get list of supported event types.

    Returns:
        List of event type identifiers
    """
    return list(EVENT_TYPE_DEFINITIONS.keys())


def get_event_type_description(event_type: str) -> Optional[str]:
    """
    Get description for a specific event type.

    Args:
        event_type: Event type identifier

    Returns:
        Event type description or None if not found
    """
    return EVENT_TYPE_DEFINITIONS.get(event_type)


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    # Test prompt building
    test_text = "President Biden met with Israeli PM Netanyahu in Washington yesterday to discuss regional security."
    test_context = {
        "cleaned_title": "Biden-Netanyahu Meeting",
        "cleaned_publication_date": "2024-01-15T10:00:00Z",
        "cleaned_author": "Jane Reporter"
    }

    print("=" * 80)
    print("TEST: Build prompt for diplomatic_relations domain")
    print("=" * 80)
    prompt = build_prompt(test_text, context=test_context, domain="diplomatic_relations", include_examples=True)
    print(prompt)
    print("\n")

    print("=" * 80)
    print("TEST: Parse LLM output")
    print("=" * 80)

    # Simulate LLM output
    llm_output = """```json
{
  "events": [
    {
      "event_type": "contact_meet",
      "trigger": {"text": "met", "start_char": 16, "end_char": 19},
      "arguments": [
        {"role": "agent", "text": "President Biden", "start_char": 0, "end_char": 15, "type": "PER"},
        {"role": "patient", "text": "Israeli PM Netanyahu", "start_char": 25, "end_char": 45, "type": "PER"},
        {"role": "place", "text": "Washington", "start_char": 49, "end_char": 59, "type": "GPE"},
        {"role": "time", "text": "yesterday", "start_char": 60, "end_char": 69, "type": "DATE"}
      ],
      "domain": "diplomatic_relations",
      "sentiment": "neutral",
      "causality": "Scheduled bilateral meeting"
    }
  ]
}
```"""

    events = parse_llm_output(llm_output, document_id="test_doc_001", original_text=test_text)

    print(f"Parsed {len(events)} events:")
    for event in events:
        print(f"\n- Event: {event.event_type}")
        print(f"  Trigger: {event.trigger.text} [{event.trigger.start_char}:{event.trigger.end_char}]")
        print(f"  Domain: {event.domain}")
        print(f"  Arguments: {len(event.arguments)}")
        for arg in event.arguments:
            print(f"    - {arg.argument_role}: {arg.entity.text} ({arg.entity.type})")

    print("\n" + "=" * 80)
    print("TEST: List supported domains and event types")
    print("=" * 80)
    print(f"Supported domains ({len(list_supported_domains())}):")
    for domain in list_supported_domains():
        print(f"  - {domain}")

    print(f"\nSupported event types ({len(list_supported_event_types())}):")
    for event_type in list_supported_event_types():
        print(f"  - {event_type}: {get_event_type_description(event_type)}")

    print("\n✓ All tests completed successfully!")
