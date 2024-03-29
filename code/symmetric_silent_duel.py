from collections import deque

from sympy import Integral
from sympy import Lambda
from sympy import Symbol
from sympy import diff
from sympy.solvers import solve
from sympy.functions.elementary.miscellaneous import Max


def f_star(prob_fun, prob_fun_var, larger_transition_times):
    '''Compute f^* as in Restrepo '57.

    In this implementation, we're working in the simplified example
    where P = Q (both players probabilities of success are the same).
    '''
    x = prob_fun_var
    P = prob_fun

    product = 1
    for a in larger_transition_times:
        product *= (1 - P(a))

    return product * diff(P(x), x) / P(x)**3


def compute_a_n(prob_fun, alpha=0):
    P = prob_fun
    t = Symbol('t0', positive=True)
    a_n = Symbol('a_n', positive=True)

    a_n_integral = Integral(
        ((1 + alpha) - (1 - alpha) * P(t)) * f_star(P, t, []), (t, a_n, 1))
    a_n_integrated = a_n_integral.doit()
    P_a_n_solutions = solve(a_n_integrated - 2 * (1 - alpha), P(a_n))
    P_a_n = Max(*P_a_n_solutions)
    print("P(a_n) = %s" % P_a_n)

    a_n_solutions = solve(P(a_n) - P_a_n, a_n)
    a_n_solutions_in_range = [soln for soln in a_n_solutions if 0 < soln <= 1]
    assert len(a_n_solutions_in_range) == 1
    a_n = a_n_solutions_in_range[0]
    print("a_n = %s" % a_n)

    h_n_integral = Integral(f_star(P, t, []), (t, a_n, 1))
    h_n_integrated = h_n_integral.doit()
    h_n = (1 - alpha) / h_n_integrated
    print("h_n = %s" % h_n)

    return (a_n, h_n)


def compute_as_and_bs(prob_fun, n, alpha=0):
    '''Compute the a's and b's for the symmetric silent duel.'''
    P = prob_fun
    t = Symbol('t0', positive=True)

    a_n, h_n = compute_a_n(prob_fun, alpha=alpha)

    normalizing_constants = deque([h_n])
    transitions = deque([a_n])
    f_star_products = deque([1, 1 - P(a_n)])

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

        print("dF_{n-%d} coeff = %s" % (step + 1, next_h * f_star_products[-1]))

        f_star_products.append(f_star_products[-1] * (1 - P_next_a))
        transitions.appendleft(next_a_soln)
        normalizing_constants.appendleft(next_h)

    return transitions


x = Symbol('x')
# compute_as_and_bs(Lambda((x,), x), 3)

compute_as_and_bs(Lambda((x,), x**3), 3)
