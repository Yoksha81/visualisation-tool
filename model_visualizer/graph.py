"""Jednostavne strukture podataka koje predstavljaju model i njegov trenutni prikaz."""
from dataclasses import dataclass, field

@dataclass
class OperationNode:
    id: str
    op: str
    target: str
    scope: str
    inputs: list[str]
    module_type: str | None

@dataclass
class GraphEdge:
    source: str
    target: str

@dataclass
class ModuleBlock:
    path: str
    module_type: str
    parent: str | None
    module_source: str = ''
    children: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)

@dataclass
class ModelGraph:
    operations: dict[str, OperationNode]
    modules: dict[str, ModuleBlock]
    edges: list[GraphEdge]

@dataclass
class VisibleNode:
    id: str
    label: str
    node_kind: str
    subtype: str | None
    expandable: bool

@dataclass
class VisibleGroup:
    id: str
    label: str
    module_type: str
    nodes: list[str]

@dataclass
class VisibleGraph:
    nodes: dict[str, VisibleNode]
    edges: list[GraphEdge]
    groups: dict[str, VisibleGroup]
