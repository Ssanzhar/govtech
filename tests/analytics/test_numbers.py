"""TDD tests for `qorgan.analytics.numbers` -- phone-number co-occurrence linking.

Convention: incidents that share at least one phone number belong to the same scam
"organization" (transitively). Everything here is pure graph logic -- no I/O.
"""

from __future__ import annotations


from qorgan.analytics.numbers import link_by_shared_numbers, numbers_for_group

# --- link_by_shared_numbers -------------------------------------------------------------


def test_empty_input_returns_empty_list():
    assert link_by_shared_numbers({}) == []


def test_two_incidents_sharing_a_number_form_one_component():
    incident_numbers = {"a": ["+7700"], "b": ["+7700"]}

    components = link_by_shared_numbers(incident_numbers)

    assert components == [frozenset({"a", "b"})]


def test_transitive_linking_across_three_incidents():
    incident_numbers = {
        "a": ["n1"],
        "b": ["n1", "n2"],
        "c": ["n2"],
    }

    components = link_by_shared_numbers(incident_numbers)

    assert components == [frozenset({"a", "b", "c"})]


def test_disjoint_incidents_are_separate_singletons():
    incident_numbers = {"a": ["n1"], "b": ["n2"]}

    components = link_by_shared_numbers(incident_numbers)

    assert frozenset({"a"}) in components
    assert frozenset({"b"}) in components
    assert len(components) == 2


def test_incident_with_empty_number_list_is_its_own_singleton():
    incident_numbers = {"a": ["n1"], "b": ["n1"], "c": []}

    components = link_by_shared_numbers(incident_numbers)

    assert frozenset({"c"}) in components
    assert len(components) == 2


def test_result_ordered_largest_first_then_lexicographically_smallest_id():
    incident_numbers = {
        "z": ["n1"],
        "y": ["n1"],
        "b": [],
        "a": [],
    }

    components = link_by_shared_numbers(incident_numbers)

    # {y, z} is the larger component (size 2) and must come first.
    assert components[0] == frozenset({"y", "z"})
    # singleton "a" sorts before singleton "b" (tie-break: smallest id).
    assert components[1] == frozenset({"a"})
    assert components[2] == frozenset({"b"})


def test_ordering_tie_break_uses_smallest_id_within_equal_sized_components():
    incident_numbers = {
        "c": ["n2"],
        "d": ["n2"],
        "a": ["n1"],
        "b": ["n1"],
    }

    components = link_by_shared_numbers(incident_numbers)

    assert components == [frozenset({"a", "b"}), frozenset({"c", "d"})]


def test_all_incidents_share_one_number_forms_single_component():
    incident_numbers = {"a": ["n1"], "b": ["n1"], "c": ["n1"]}

    components = link_by_shared_numbers(incident_numbers)

    assert components == [frozenset({"a", "b", "c"})]


def test_covers_every_incident_id_exactly_once():
    incident_numbers = {"a": ["n1"], "b": ["n1"], "c": ["n2"], "d": []}

    components = link_by_shared_numbers(incident_numbers)

    covered = frozenset().union(*components)
    assert covered == frozenset(incident_numbers)
    # No id appears twice across components.
    assert sum(len(c) for c in components) == len(incident_numbers)


# --- numbers_for_group -------------------------------------------------------------------


def test_numbers_for_group_returns_sorted_distinct_numbers():
    incident_numbers = {"a": ["n2", "n1"], "b": ["n1", "n3"]}

    result = numbers_for_group(["a", "b"], incident_numbers)

    assert result == ("n1", "n2", "n3")


def test_numbers_for_group_ignores_ids_not_in_mapping():
    incident_numbers = {"a": ["n1"]}

    result = numbers_for_group(["a", "unknown"], incident_numbers)

    assert result == ("n1",)


def test_numbers_for_group_empty_group_returns_empty_tuple():
    assert numbers_for_group([], {"a": ["n1"]}) == ()


def test_numbers_for_group_empty_mapping_returns_empty_tuple():
    assert numbers_for_group(["a"], {}) == ()


def test_numbers_for_group_returns_tuple_type():
    result = numbers_for_group(["a"], {"a": ["n1"]})
    assert isinstance(result, tuple)
