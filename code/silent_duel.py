from collections import deque
from dataclasses import dataclass
from typing import NewType

from sympy import Integral
from sympy import Lambda
from sympy import N
from sympy import Symbol
from sympy import diff
from sympy.functions.elementary.miscellaneous import Max
from sympy.solvers import solve


SuccessFn = NewType('SuccessFn', Lambda)


@dataclass
class SilentDuelInput:
    '''Class representing the static input data to the Silent Duel
    problem.'''
    player_1_action_count: int

    player_2_action_count: int

    player_1_action_success: SuccessFn

    player_2_action_success: SuccessFn


def f_star(
        player_action_success,
        opponent_action_success,
        variable,
        larger_transition_times):
    '''Compute f^* as in Restrepo '57.

    The inputs can be chosen so that the appropriate f^* is built
    for either player. I.e., if we want to compute f^* for player 1,
    player_action_succes should correspond to P, opponent_action_success
    to Q, and larger_transition_times to the b_j.

    If the inputs are switched appropriately, f^* is computed for player 2.
    '''
    P = player_action_success
    Q = opponent_action_success

    product = 1
    for a in larger_transition_times:
        product *= (1 - P(a))

    return (
        product * diff(Q(variable), variable) / (Q(variable)**2 * P(variable))
    )


def compute_a_n_and_b_m(silent_duel_input, alpha=0, beta=0):
    P = silent_duel_input.player_1_action_success
    Q = silent_duel_input.player_2_action_success
    t = Symbol('t0', positive=True)
    a_n = Symbol('a_n', positive=True)
    b_m = Symbol('b_m', positive=True)

    p1_fstar = f_star(P, Q, t, [])
    p2_fstar = f_star(Q, P, t, [])

    # the a_n part
    a_n_integral = Integral(
        ((1 + alpha) - (1 - alpha) * P(t)) * p1_fstar, (t, a_n, 1))
    a_n_integrated = a_n_integral.doit()
    P_a_n_solutions = solve(a_n_integrated - 2 * (1 - alpha), P(a_n))
    P_a_n_solutions = [x for x in P_a_n_solutions if x.is_real]
    P_a_n = Max(*P_a_n_solutions)
    print("P(a_n) = %s" % P_a_n)

    a_n_solutions = solve(P(a_n) - P_a_n, a_n)
    a_n_solutions_in_range = [soln for soln in a_n_solutions if 0 < soln <= 1]
    assert len(a_n_solutions_in_range) == 1
    a_n = a_n_solutions_in_range[0]
    print("a_n = %s" % a_n)

    # the b_m part
    b_m_integral = Integral(
        ((1 + beta) - (1 - beta) * Q(t)) * p2_fstar, (t, b_m, 1))
    b_m_integrated = b_m_integral.doit()
    Q_b_m_solutions = solve(b_m_integrated - 2 * (1 - beta), Q(b_m))
    Q_b_m_solutions = [x for x in Q_b_m_solutions if x.is_real]
    Q_b_m = Max(*Q_b_m_solutions)
    print("Q(b_m) = %s" % Q_b_m)

    b_m_solutions = solve(Q(b_m) - Q_b_m, b_m)
    b_m_solutions_in_range = [soln for soln in b_m_solutions if 0 < soln <= 1]
    assert len(b_m_solutions_in_range) == 1
    b_m = b_m_solutions_in_range[0]
    print("b_m = %s" % b_m)

    # the h part
    h_n_integral = Integral(p1_fstar, (t, a_n, 1))
    h_n_integrated = h_n_integral.doit()
    h_n = (1 - alpha) / h_n_integrated
    print("h_n = %s" % h_n)

    # the k part
    k_m_integral = Integral(p2_fstar, (t, b_m, 1))
    k_m_integrated = k_m_integral.doit()
    k_m = (1 - beta) / k_m_integrated
    print("k_m = %s" % k_m)

    return (a_n, b_m, h_n, k_m)


def compute_as_and_bs(duel_input, alpha=0, beta=0):
    '''Compute the a's and b's for the silent duel.'''
    P = prob_fun
    t = Symbol('t0', positive=True)

    a_n, h_n = compute_a_n(prob_fun, alpha=alpha)

    normalizing_constants = deque([h_n])
    transitions = deque([a_n])

    for step in range(n):
        # prepending new a's and h's to the front of the list
        last_a = transitions[0]
        last_h = normalizing_constants[0]
        next_a = Symbol('a', positive=True)

        next_a_integral = Integral(
            (1 - P(t)) * f_star(P, t, transitions), (t, next_a, last_a))
        next_a_integrated = next_a_integral.doit()
        # print("%s" % next_a_integrated)
        P_next_a_solutions = solve(next_a_integrated - 1 / last_h, P(next_a))
        print("P(a_{n-%d}) is one of %s" % (step + 1, P_next_a_solutions))
        P_next_a = Max(*P_next_a_solutions)

        next_a_solutions = solve(P(next_a) - P_next_a, next_a)
        next_a_solutions_in_range = [
            soln for soln in next_a_solutions if 0 < soln <= 1]
        assert len(next_a_solutions_in_range) == 1
        next_a_soln = next_a_solutions_in_range[0]
        print("a_{n-%d} = %s" % (step + 1, next_a_soln))

        next_h_integral = Integral(
            f_star(P, t, transitions), (t, next_a_soln, last_a))
        next_h = 1 / next_h_integral.doit()
        print("h_{n-%d} = %s" % (step + 1, next_h))

        transitions.appendleft(next_a_soln)
        normalizing_constants.appendleft(next_h)

    return transitions


x = Symbol('x')
P = Lambda((x,), x)
Q = Lambda((x,), x**2)
duel_input = SilentDuelInput(
    player_1_action_count=3,
    player_2_action_count=3,
    player_1_action_success=P,
    player_2_action_success=P,
)
compute_a_n_and_b_m(duel_input)
# compute_as_and_bs(Lambda((x,), x), 10)
