'''
A module that isolates the computation of f^*
'''

from typing import NewType
from typing import Iterable
import random

from sympy import Expr
from sympy import Lambda
from sympy import Piecewise
from sympy import Symbol
from sympy import diff

SuccessFn = NewType('SuccessFn', Lambda)


def simple_f_star(player_action_success: SuccessFn,
                  opponent_action_success: SuccessFn,
                  variable: Symbol,
                  larger_transition_times: Iterable[float]) -> Expr:
    P: SuccessFn = player_action_success
    Q: SuccessFn = opponent_action_success

    non_product_term = diff(Q(variable), variable) / (Q(variable)**2 * P(variable))

    product = 1
    for b in larger_transition_times:
        product *= (1 - Q(b))

    return product * non_product_term


def f_star(player_action_success: SuccessFn,
           opponent_action_success: SuccessFn,
           variable: Symbol,
           opponent_transition_times: Iterable[float],
           scale_by: Expr = 1) -> Expr:
    '''Compute f^* as in Restrepo '57.

    The inputs can be chosen so that the appropriate f^* is built
    for either player. I.e., if we want to compute f^* for player 1,
    player_action_success should correspond to P, opponent_action_success
    to Q, and larger_transition_times to the b_j.

    If the inputs are switched appropriately, f^* is computed for player 2.
    '''
    P: SuccessFn = player_action_success
    Q: SuccessFn = opponent_action_success

    '''
    We compute f^* as a Piecewise function of the following form:

    [prod_{i=1}^m (1-Q(b_i))] * Q'(t) / Q^2(t)P(t)    if t < b_1
    [prod_{i=2}^m (1-Q(b_i))] * Q'(t) / Q^2(t)P(t)    if t < b_2
    [prod_{i=3}^m (1-Q(b_i))] * Q'(t) / Q^2(t)P(t)    if t < b_3
             .
             .
             .
    [1] *  Q'(t) / Q^2(t) P(t)                        if t >= b_m
    '''
    non_product_term = (
        scale_by * diff(Q(variable), variable) / (Q(variable)**2 * P(variable))
    )

    piecewise_components = []
    for i, b_j in enumerate(opponent_transition_times):
        larger_transition_times = opponent_transition_times[i:]

        product = 1
        for b in larger_transition_times:
            product *= (1 - Q(b))

        term = product * non_product_term
        piecewise_components.append((term, variable < b_j))

    # last term is when there are no larger transition times.
    piecewise_components.append((non_product_term, True))

    return Piecewise(*piecewise_components)
