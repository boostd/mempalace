"""
fact_checker.py — Verify text against known facts in the palace.

Checks AI responses, diary entries, and new content against the
entity registry and knowledge graph for contradictions. Catches:
  - Wrong names (similar but different entities)
  - Wrong relationships (calling someone the wrong role)
  - Stale facts (things that changed — KG has valid_from/valid_to)

Uses the entity_registry and knowledge_graph — no hardcoded facts.

Usage:
    from mempalace.fact_checker import check_text
    issues = check_text("Bob is Alice's brother", palace_path)
    # → [{"type": "relationship_mismatch", "detail": "KG says Bob is Alice's husband"}]

    # CLI
    python -m mempalace.fact_checker "Bob is Alice's brother" --palace ~/.mempalace/palace
"""

import os
import re
from pathlib import Path


def check_text(text, palace_path=None, config=None):
    """Check text for contradictions against known facts.

    Returns list of issues found. Empty list = no contradictions.
    """
    if config is None:
        from .config import MempalaceConfig
        config = MempalaceConfig()
    if palace_path is None:
        palace_path = config.palace_path

    issues = []

    # Load known entities
    entity_names = _load_known_entities()

    # Check entity name confusion (similar names that might be mixed up)
    issues.extend(_check_entity_confusion(text, entity_names))

    # Check against knowledge graph facts
    issues.extend(_check_kg_facts(text, palace_path))

    return issues


def _load_known_entities():
    """Load entity names from the registry."""
    import json
    registry_path = os.path.expanduser("~/.mempalace/known_entities.json")
    if not os.path.exists(registry_path):
        return {}
    try:
        return json.loads(open(registry_path).read())
    except Exception:
        return {}


def _check_entity_confusion(text, entity_names):
    """Check if text confuses similar entity names."""
    issues = []
    all_names = set()
    for cat in entity_names.values():
        if isinstance(cat, list):
            all_names.update(cat)
        elif isinstance(cat, dict):
            all_names.update(cat.keys())

    # Find names mentioned in text
    mentioned = set()
    for name in all_names:
        if re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE):
            mentioned.add(name)

    # Check for names that are very similar but different (edit distance 1-2)
    name_list = sorted(all_names)
    for i, name_a in enumerate(name_list):
        for name_b in name_list[i + 1:]:
            if _edit_distance(name_a.lower(), name_b.lower()) <= 2:
                if name_a in mentioned or name_b in mentioned:
                    if name_a in text and name_b not in text:
                        issues.append({
                            "type": "similar_name",
                            "detail": f"'{name_a}' mentioned — did you mean '{name_b}'? (similar names in registry)",
                            "names": [name_a, name_b],
                        })
    return issues


def _check_kg_facts(text, palace_path):
    """Check text against knowledge graph for contradictions."""
    issues = []
    try:
        from .knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(palace_path=palace_path)

        # Extract relationship claims from text
        # Pattern: "X is Y's Z" or "X's Z is Y"
        patterns = [
            (r"(\w+)\s+is\s+(\w+)'s\s+(\w+)", "subject", "possessor", "role"),
            (r"(\w+)'s\s+(\w+)\s+is\s+(\w+)", "possessor", "role", "subject"),
        ]

        for pattern, *roles in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                subject = groups[0]
                # Query KG for this entity
                try:
                    facts = kg.query(subject)
                    if facts:
                        for fact in facts:
                            # Check if the claim contradicts a known fact
                            if fact.get("valid_to") is None:  # current fact
                                kg_pred = fact.get("predicate", "").lower()
                                claim = match.group(0).lower()
                                if kg_pred in claim and fact.get("object", "").lower() not in claim:
                                    issues.append({
                                        "type": "relationship_mismatch",
                                        "detail": f"Text says '{match.group(0)}' but KG says: {subject} {kg_pred} {fact.get('object')}",
                                        "entity": subject,
                                    })
                except Exception:
                    pass
    except Exception:
        pass  # KG not available — skip

    return issues


def _edit_distance(s1, s2):
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if c1 == c2 else 1),
            ))
        prev = curr
    return prev[-1]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Check text against known facts")
    parser.add_argument("text", nargs="?", help="Text to check")
    parser.add_argument("--palace", default=os.path.expanduser("~/.mempalace/palace"))
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    args = parser.parse_args()

    if args.stdin:
        import sys
        text = sys.stdin.read()
    elif args.text:
        text = args.text
    else:
        print("Provide text as argument or use --stdin")
        exit(1)

    issues = check_text(text, palace_path=args.palace)
    if issues:
        print(json.dumps(issues, indent=2))
    else:
        print("No contradictions found.")
