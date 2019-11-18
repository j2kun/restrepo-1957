import math
from assertpy import assert_that
from silent_duel import *

from sympy import Expr
from sympy import Lambda
from sympy import N
from sympy import Symbol


EPSILON = 1e-6


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


def test_symmetric_duel_output_distributions():
    expected_transitions = [
        1 / 7,
        1 / 5,
        1 / 3,
        1,
    ]
    expected_df_coeffs = [
        1 / 12,
        1 / 8,
        1 / 4,
    ]

    x = Symbol('x')
    P = Lambda((x,), x)
    Q = Lambda((x,), x)

    duel_input = SilentDuelInput(
        player_1_action_count=3,
        player_2_action_count=3,
        player_1_action_success=P,
        player_2_action_success=Q,
    )
    player_strategies = optimal_strategies(duel_input)
    p1_strategy = player_strategies.p1_strategy
    p2_strategy = player_strategies.p2_strategy

    assert_that(p1_strategy).is_equal_to(p2_strategy)
    for transition_time, expected_transition_time in zip(p1_strategy.transition_times(), expected_transitions):
        assert_that(transition_time).is_close_to(expected_transition_time, 1e-3)

    for i in range(3):
        print(i)
        actual_ad = p1_strategy.action_distributions[i]
        assert_that(actual_ad.support_start).is_close_to(expected_transitions[i], 1e-3)
        assert_that(actual_ad.support_end).is_close_to(expected_transitions[i + 1], 1e-3)
        assert_that(actual_ad.point_mass).is_equal_to(0)

        cdf = actual_ad.cumulative_density_function.expr
        expr_intervals = cdf.as_expr_set_pairs()
        for (j, (expr, interval)) in enumerate(expr_intervals):
            if j == 0:
                assert_that(float(expr)).is_equal_to(0.0)
            elif j == len(expr_intervals) - 1:
                assert_that(float(expr)).is_equal_to(1.0)
            else:
                if isinstance(expr, Lambda):
                    expr = expr.expr
                t = list(expr.free_symbols)[0]
                expected_df = expected_df_coeffs[i] / t ** 3
                assert_that(expr.diff(t)).is_equal_to(expected_df)
                assert_that(float(interval.start)).is_close_to(expected_transitions[i], EPSILON)
                assert_that(float(interval.end)).is_close_to(expected_transitions[i + 1], EPSILON)


def test_asymmetric_duel_one_action_proper_alpha_beta():
    x = Symbol('x')
    P = Lambda((x,), x)
    Q = Lambda((x,), x**2)

    duel_input = SilentDuelInput(
        player_1_action_count=1,
        player_2_action_count=1,
        player_1_action_success=P,
        player_2_action_success=Q,
    )

    output = optimal_strategies(duel_input)

    assert_that(output.p1_strategy.action_distributions).is_length(1)
    assert_that(output.p2_strategy.action_distributions).is_length(1)

    p1_dist = output.p1_strategy.action_distributions[0]
    p2_dist = output.p2_strategy.action_distributions[0]

    assert_that(p1_dist.support_start).is_close_to(0.481, 1e-3)
    assert_that(p2_dist.support_start).is_close_to(0.481, 1e-3)
    assert_that(p2_dist.point_mass).is_close_to(0.0727, 1e-3)
    output.validate(err_on_fail=True)


'''
def test_asymmetric_duel_action_counts_differ():
    x = Symbol('x')
    P = Lambda((x,), x)
    Q = Lambda((x,), x**2)

    duel_input = SilentDuelInput(
        player_1_action_count=3,
        player_2_action_count=4,
        player_1_action_success=P,
        player_2_action_success=Q,
    )

    output = optimal_strategies(duel_input)

    assert_that(output.p1_strategy.action_distributions).is_length(3)
    assert_that(output.p2_strategy.action_distributions).is_length(4)
'''
