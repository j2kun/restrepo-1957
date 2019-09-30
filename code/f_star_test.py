import math
from assertpy import assert_that
from f_star import *

from sympy import Expr
from sympy import Lambda
from sympy import N
from sympy import Symbol


EPSILON = 1e-6


def assert_eq(expr, val):
    assert_that(float(expr)).is_close_to(val, EPSILON)


def test_three_transitions():
    x = Symbol('x')
    P = Lambda((x,), x)
    Q = Lambda((x,), x)

    opponent_transition_times = [0.25, 0.5, 0.75]

    F = Lambda((x,), f_star(P, Q, x, opponent_transition_times))
    assert_eq(F(1), 1)
    assert_eq(F(0.8), 1 / 0.8 ** 3)
    assert_eq(F(0.6), (1 / 4) / 0.6 ** 3)
    assert_eq(F(0.4), (1 / 8) / 0.4 ** 3)
    # (1-1/4)*(1-1/2)*(1-3/4)
    assert_eq(F(0.2), (3 / 32) / 0.2 ** 3)


def test_empty():
    x = Symbol('x')
    P = Lambda((x,), x)
    Q = Lambda((x,), x)

    opponent_transition_times = []

    F = Lambda((x,), f_star(P, Q, x, opponent_transition_times))
    assert_eq(F(1), 1)
    assert_eq(F(0.3), 1 / 0.3**3)
