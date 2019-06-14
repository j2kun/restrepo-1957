from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import List
from typing import NewType
import random

from sympy import Expr
from sympy import Integral
from sympy import Lambda
from sympy import N
from sympy import Symbol
from sympy import diff
from sympy.functions.elementary.miscellaneous import Max
from sympy.solvers import solve


SuccessFn = NewType('SuccessFn', Lambda)


def solve_unique_real(expr, var, solution_min=0, solution_max=1):
    '''
    Solve an equation and return any solution in the given range. This should
    be used when the caller knows that there is a unique solution to the
    equation in the given range.

    If the range bounds are set to None, then the assumption is that there is a
    unique global real solution. By default the range bound is [0,1].
    '''
    # print("solving {} = 0 for {}".format(expr, var))
    solutions = solve(expr, var)
    solutions = [x for x in solutions if x.is_real]
    if solution_min is not None:
        solutions = [x for x in solutions if x >= solution_min]
    if solution_max is not None:
        solutions = [x for x in solutions if x <= solution_max]

    assert len(solutions) == 1, "Expected unique solution but found {}".format(solutions)
    return solutions[0]


@dataclass
class SilentDuelInput:
    '''Class representing the static input data to the Silent Duel
    problem.'''
    player_1_action_count: int

    player_2_action_count: int

    player_1_action_success: SuccessFn

    player_2_action_success: SuccessFn


@dataclass
class ActionDistribution:
    '''The interval on which this distribution occurs.'''
    support_start: float
    support_end: float

    '''The cumulative density function for the distribution.'''
    cumulative_density_function: Expr

    def sample(self):
        uniform_random_number = random.random()
        x = Symbol('x', positive=True)
        return solve_unique_real(
            self.cumulative_density_function(x) - uniform_random_number,
            x,
            solution_min=self.support_start,
            solution_max=self.support_end,
        )


@dataclass
class SilentDuelOutput:
    '''
    The distribution functions representing the probability of taking each
    action.
    '''
    action_distributions: List[ActionDistribution]

    def sample_game_strategy(self):
        return [d.sample() for d in self.action_distributions]

    '''
    Returns the ordered list of action transition times for this player.
    '''

    def transition_times(self):
        times = [self.action_distributions[0].support_start]
        times.extend([d.support_end for d in self.action_distributions])
        return times


@dataclass
class IntermediateState:
    '''Class representing the intermediate state of the silent
    duel construction algorithm. This class is mutated as the
    algorithm progresses.'''

    '''
    A list of the transition times compute thus far. This field
    maintains the invariant of being sorted. Thus, the first element
    in the list is a_{i + 1}, the most recently computed value of
    player 1's transition times, and the last element is a_{n + 1} = 1.
    This value is set on initializtion with `new`.
    '''
    player_1_transition_times: List[Expr]

    '''
    Same as player_1_transition_times, but for player 2 with b_j
    and b_m.
    '''
    player_2_transition_times: List[Expr]

    '''
    The values of h_i, normalizing constants for the action
    probability distributions for player 1. Has the same sorting
    invariant as the transition time lists.
    '''
    player_1_normalizing_constants: List[Expr]

    '''
    Same as player_1_normalizing_constants, but for player 2,
    i.e., the k_j normalizing constants.
    '''
    player_2_normalizing_constants: List[Expr]

    @staticmethod
    def new():
        '''
        Create a new state object, and set a_{n+1} = 1, b_{m+1}=1.
        '''
        return IntermediateState(
            player_1_transition_times=deque([1]),
            player_2_transition_times=deque([1]),
            player_1_normalizing_constants=deque([]),
            player_2_normalizing_constants=deque([]),
        )

    def add_p1(self, transition_time, normalizing_constant):
        self.player_1_transition_times.appendleft(transition_time)
        self.player_1_normalizing_constants.appendleft(normalizing_constant)

    def add_p2(self, transition_time, normalizing_constant):
        self.player_2_transition_times.appendleft(transition_time)
        self.player_2_normalizing_constants.appendleft(normalizing_constant)


"""
The rest  of this file contains functions constructing the optimal strategy
for each player.
"""


def f_star(player_action_success: SuccessFn,
           opponent_action_success: SuccessFn,
           variable: Symbol,
           larger_transition_times: List[float]) -> Expr:
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


def compute_ai_and_bj(duel_input: SilentDuelInput,
                      intermediate_state: IntermediateState,
                      alpha: float = 0,
                      beta: float = 0):
    '''
    Compute a pair of a_i and b_j transition times for both players,
    using the intermediate state of the algorithm computed so far.

    This function also computes a_n and b_m when the intermediate_state
    input has no larger transition times for the opposite player. In
    those cases, the integrals and equations being solved are slightly
    different; they include some terms involving alpha and beta. In all
    other cases, the alpha and beta parameters are unused.
    '''
    P = duel_input.player_1_action_success
    Q = duel_input.player_2_action_success
    t = Symbol('t0', positive=True)
    a_i = Symbol('a_i', positive=True)
    b_j = Symbol('b_j', positive=True)

    p1_transitions = intermediate_state.player_1_transition_times
    p2_transitions = intermediate_state.player_2_transition_times

    # the left end of the transitions arrays contain the smallest
    # (latest computed) transition time for each player.
    # these are set to 1 for an empty intermediate state, i.e. for a_n, b_m
    a_i_plus_one = p1_transitions[0]
    b_j_plus_one = p2_transitions[0]
    computing_a_n = a_i_plus_one == 1
    computing_b_m = b_j_plus_one == 1

    # the a_i part
    if computing_a_n:
        p1_fstar = f_star(P, Q, t, [])
        p1_integrand = ((1 + alpha) - (1 - alpha) * P(t)) * p1_fstar
        p1_integral_target = 2 * (1 - alpha)
    else:
        p1_fstar_parameters = list(p2_transitions)[:-1]  # ignore b_{m+1} = 1
        p1_fstar = f_star(P, Q, t, p1_fstar_parameters)
        p1_integrand = (1 - P(t)) * p1_fstar
        p1_integral_target = 1 / intermediate_state.player_1_normalizing_constants[0]

    a_i_integral = Integral(p1_integrand, (t, a_i, a_i_plus_one))
    a_i_integrated = a_i_integral.doit()
    a_i = solve_unique_real(
        a_i_integrated - p1_integral_target,
        a_i,
        solution_min=0,
        solution_max=a_i_plus_one
    )
    # print("a_i = %s" % a_i)

    # the b_j part
    if computing_b_m:
        p2_fstar = f_star(Q, P, t, [])
        p2_integrand = ((1 + beta) - (1 - beta) * P(t)) * p2_fstar
        p2_integral_target = 2 * (1 - beta)
    else:
        p2_fstar_parameters = list(p1_transitions)[:-1]  # ignore a_{n+1} = 1
        p2_fstar = f_star(Q, P, t, p2_fstar_parameters)
        p2_integrand = (1 - P(t)) * p2_fstar
        p2_integral_target = 1 / intermediate_state.player_2_normalizing_constants[0]

    b_j_integral = Integral(p2_integrand, (t, b_j, b_j_plus_one))
    b_j_integrated = b_j_integral.doit()
    b_j = solve_unique_real(
        b_j_integrated - p2_integral_target,
        b_j,
        solution_min=0,
        solution_max=b_j_plus_one
    )
    # print("b_j = %s" % b_j)

    # the h_i part
    h_i_integral = Integral(p1_fstar, (t, a_i, a_i_plus_one))
    h_i_integrated = h_i_integral.doit()
    h_i_numerator = (1 - alpha) if computing_a_n else 1
    h_i = h_i_numerator / h_i_integrated
    # print("h_i = %s" % h_i)

    # the k_j part
    k_j_integral = Integral(p2_fstar, (t, b_j, b_j_plus_one))
    k_j_integrated = k_j_integral.doit()
    k_j_numerator = (1 - beta) if computing_b_m else 1
    k_j = k_j_numerator / k_j_integrated
    # print("k_j = %s" % k_j)

    return (a_i, b_j, h_i, k_j)


def compute_as_and_bs(duel_input: SilentDuelInput,
                      alpha: float = 0,
                      beta: float = 0) -> IntermediateState:
    '''
    Compute the a's and b's for the silent duel, given a fixed
    alpha and beta as input.
    '''
    t = Symbol('t0', positive=True)

    p1_index = duel_input.player_1_action_count
    p2_index = duel_input.player_2_action_count
    intermediate_state = IntermediateState.new()

    while p1_index > 0 or p2_index > 0:
        # compute a new a_i and b_j, keeping the larger as the next parameter
        (a_i, b_j, h_i, k_j) = compute_ai_and_bj(duel_input, intermediate_state)

        # the larger of a_n, b_m is kept as a parameter, then the other will be repeated
        # in the next iteration; e.g., a_{n-1} and b_m (the latter using a_n in its f*)
        # possibly add both if a_i == b_j
        if a_i >= b_j and p1_index > 0:
            intermediate_state.add_p1(N(a_i), N(h_i))
            print("a_{} = {}, h_{} = {}".format(p1_index, N(a_i), p1_index, N(h_i)))
            p1_index -= 1
        if b_j >= a_i and p2_index > 0:
            intermediate_state.add_p2(N(b_j), N(k_j))
            print("b_{} = {}, k_{} = {}".format(p2_index, N(b_j), p2_index, N(k_j)))
            p2_index -= 1

    return intermediate_state


x = Symbol('x')
P = Lambda((x,), x)
Q = Lambda((x,), x**2)

duel_input = SilentDuelInput(
    player_1_action_count=3,
    player_2_action_count=3,
    player_1_action_success=P,
    player_2_action_success=P,
)
compute_as_and_bs(duel_input, alpha=0, beta=0)

# next: set up tests for symmetric case, implement binary search for alpha, beta.

f = ActionDistribution(support_start=0, support_end=1, cumulative_density_function=Q)
