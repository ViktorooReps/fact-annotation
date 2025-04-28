import copy
import uuid
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional


@dataclass
class Entity:
    name: str
    type: str
    id: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    thumbnail: Optional[str] = None
    aliases: Optional[list[str]] = None

    def has_name(self) -> bool:
        return len(self.name) > 0 and self.name != 'No name'

    def __post_init__(self):
        self.name = self.name.strip()

        if self.id is None:
            self.id = str(uuid.uuid4())

        if self.aliases is None:
            self.aliases = [self.name]

        if self.name not in self.aliases:
            self.aliases.append(self.name)

        self.aliases = [alias.strip() for alias in self.aliases]
        self.aliases = sorted(self.aliases)

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id

    def merge(self, other):
        assert self.id == other.id
        assert self.type == other.type

        if not self.has_name():
            self.name = other.name

        if self.url is None:
            self.url = other.url

        if self.description is None:
            self.description = other.description

        self.aliases = sorted(set(self.aliases).union(other.aliases))

        return self


def merge_entities(entities: list[Entity]) -> Optional[Entity]:
    if not len(entities):
        return None

    lead_entity = copy.deepcopy(entities[0])
    for entity in entities[1:]:
        lead_entity.merge(entity)

    return lead_entity


def disambiguate_entities(entities: list[Entity]) -> list[Entity]:
    desc2entities = defaultdict(list)
    for entity in entities:
        desc2entities[(entity.id, entity.type)].append(entity)

    entities = []
    for same_id_entities in desc2entities.values():
        merged = merge_entities(same_id_entities)
        if merged is not None:
            entities.append(merged)

    return entities


@dataclass
class BiDirectionalRelation:  # FIXME: only bi-directional for now
    type: str
    from_id: str
    to_id: str

    @property
    def _signature(self):
        return (self.from_id, self.to_id, self.type) \
            if self.from_id > self.to_id \
            else (self.to_id, self.from_id, self.type)

    def __eq__(self, other):
        if not isinstance(other, BiDirectionalRelation):
            return False
        return self._signature == other._signature

    def __lt__(self, other):
        return self._signature < other._signature

    def __hash__(self):
        return hash(self._signature)


def disambiguate_relations(relations: list[BiDirectionalRelation]) -> list[BiDirectionalRelation]:
    return list(set(relations))


def _similarity(a: str, b: str) -> float:
    """Case-insensitive similarity in the range [0‒1]."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fuzzy_find_entity_by_alias(
    search_term: str,
    global_entities: list[Entity],
    *,
    threshold: float = 0.60,
) -> list[Entity]:
    """
    Return all entities whose *name* or *aliases* fuzzily match ``search_term``.

    Parameters
    ----------
    search_term
        The user-supplied text to match.
    global_entities
        The master list of entities to search.
    threshold
        Minimum similarity (0–1).  Increase it for stricter matching.

    Returns
    -------
    list[Entity]
        Entities sorted by descending similarity, then alphabetically.
    """
    if not search_term or not global_entities:
        return []

    search_term = search_term.strip()
    scored: list[tuple[float, Entity]] = []

    for entity in global_entities:
        # Collect every candidate string we want to compare against
        candidates = [entity.name, *(entity.aliases or [])]

        # Find the best score for this entity
        best = max((_similarity(search_term, cand) for cand in candidates), default=0.0)

        if best >= threshold:
            scored.append((best, entity))

    # Sort by similarity (descending) then name for deterministic order
    scored.sort(key=lambda t: (-t[0], t[1].name.lower()))
    return [entity for _, entity in scored]
