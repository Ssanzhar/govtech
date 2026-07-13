"""Phone-number co-occurrence linking for scam "organization" grouping.

Invariant: two incidents belong to the same group iff they are connected, directly or
transitively, by at least one shared phone number. Every incident id in the input is
covered by exactly one returned component (an incident with no/unique numbers forms its
own singleton). Pure graph logic: no I/O, no mutation of inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import networkx as nx


def link_by_shared_numbers(incident_numbers: Mapping[str, Sequence[str]]) -> list[frozenset[str]]:
    """Group incident ids into connected components linked by shared phone numbers.

    Builds a graph with one node per incident id and an edge between two incidents
    whenever they share at least one phone number, then returns its connected
    components. Ordered largest-first, ties broken by the lexicographically smallest
    incident id in the component. Returns `[]` for empty input.
    """
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(incident_numbers.keys())

    number_to_incidents: dict[str, list[str]] = {}
    for incident_id, numbers in incident_numbers.items():
        for number in numbers:
            number_to_incidents.setdefault(number, []).append(incident_id)

    for incidents_sharing_number in number_to_incidents.values():
        first_incident = incidents_sharing_number[0]
        for other_incident in incidents_sharing_number[1:]:
            graph.add_edge(first_incident, other_incident)

    components = [frozenset(component) for component in nx.connected_components(graph)]
    return sorted(components, key=lambda component: (-len(component), min(component, default="")))


def numbers_for_group(
    group: Iterable[str], incident_numbers: Mapping[str, Sequence[str]]
) -> tuple[str, ...]:
    """Return the sorted tuple of distinct phone numbers used by incidents in `group`.

    Incident ids not present in `incident_numbers` are ignored.
    """
    distinct_numbers: set[str] = set()
    for incident_id in group:
        distinct_numbers.update(incident_numbers.get(incident_id, ()))
    return tuple(sorted(distinct_numbers))
