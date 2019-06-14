import math
from assertpy import assert_that
from silent_duel import *

from sympy import Expr
from sympy import Lambda
from sympy import N
from sympy import Symbol


EPSILON = 1e-3


def test_symmetric_duel_linear_action_probability():
    x = Symbol('x')
    P = Lambda((x,), x)

    duel_input = SilentDuelInput(
        player_1_action_count=11,
        player_2_action_count=11,
        player_1_action_success=P,
        player_2_action_success=P,
    )

    expected_transitions = [
        1 / 23,
        1 / 21,
        1 / 19,
        1 / 17,
        1 / 15,
        1 / 13,
        1 / 11,
        1 / 9,
        1 / 7,
        1 / 5,
        1 / 3,
        1,
    ]
    expected_normalizers = [
        88179 / 1048576,
        46189 / 524288,
        12155 / 131072,
        6435 / 65536,
        429 / 4096,
        231 / 2048,
        63 / 512,
        35 / 256,
        5 / 32,
        3 / 16,
        1 / 4,
    ]

    state = compute_as_and_bs(duel_input, alpha=0, beta=0)
    assert_that(state.player_1_transition_times).is_length(len(expected_transitions))
    assert_that(state.player_2_transition_times).is_length(len(expected_transitions))
    assert_that(state.player_1_normalizing_constants).is_length(len(expected_normalizers))
    assert_that(state.player_2_normalizing_constants).is_length(len(expected_normalizers))

    for actual, expected in zip(state.player_1_transition_times, expected_transitions):
        assert_that(actual).is_close_to(expected, EPSILON)

    for actual, expected in zip(state.player_2_transition_times, expected_transitions):
        assert_that(actual).is_close_to(expected, EPSILON)

    for actual, expected in zip(state.player_1_normalizing_constants, expected_normalizers):
        assert_that(actual).is_close_to(expected, EPSILON)

    for actual, expected in zip(state.player_2_normalizing_constants, expected_normalizers):
        assert_that(actual).is_close_to(expected, EPSILON)
