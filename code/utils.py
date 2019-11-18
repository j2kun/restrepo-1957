from sympy import Piecewise
from scipy.optimize import fsolve

from parameters import VALIDATION_EPSILON
from parameters import SOLVE_EPSILON


def solve_unique_real(expr, var, solution_min=0, solution_max=1):
    '''
    Solve an equation and return any solution in the given range. This should
    be used when the caller knows that there is a unique solution to the
    equation in the given range.

    If the range bounds are set to None, then the assumption is that there is a
    unique global real solution. By default the range bound is [0,1].
    '''

    # sympy is a headache when it comes to solving piecewise functions
    # since we only care about real values in a given range, default to
    # scipy which is much faster

    # Update: scipy fails due to the jump discontinuities. My first idea
    # for a workaround is to sample (epsilon-net) and only try to solve
    # using the piecewise component that is closest to zero
    expr = expr.simplify()
    if isinstance(expr, Piecewise):
        expr, guess = choose_best_component(expr, var, solution_min, solution_max)
    else:
        guess = (solution_min + solution_max) / 2

    def f(x):
        return expr.subs(var, x).evalf()

    solution_data = fsolve(
        f,
        guess,
        xtol=SOLVE_EPSILON,
        full_output=True,
    )

    if solution_data[2] > 3:
        print(expr)
        print(solution_data)

    solution = solution_data[0][0]
    return solution


def arange(start, stop, step):
    if step <= 0 or stop <= step:
        raise ValueError(
            "invalid params: {} {} {}".format(start, stop, step))
    while start < stop:
        yield start
        start += step


def choose_best_component(expr, var, range_min, range_max, epsilon_net=1e-2):
    def f(x):
        return expr.subs(var, x).evalf()

    tested_values = []
    for test in arange(range_min, range_max, epsilon_net):
        value = f(test)
        if value.is_real and not value.is_infinite:
            tested_values.append((test, f(test)))

    min_test_and_value = min(tested_values, key=lambda x: abs(x[1]))
    min_test = min_test_and_value[0]
    for (component, domain) in expr.as_expr_set_pairs():
        if component.is_zero:
            continue
        if min_test in domain:
            return (component, min_test)

    raise ValueError("min_test was {}".format(min_test))


def mask_piecewise(F, variable, domain_start, domain_end, before_domain_val=0, after_domain_val=0):
    '''
    Given a piecewise, add conditions (0, variable < domain_start)
    and (0, variable > domain_end)

    Assumes the piecewise given as input has conditions specified
    only by their upper bound. (very specific to this project)
    '''
    if not isinstance(F, Piecewise):
        return Piecewise(
            (before_domain_val, variable <= domain_start),
            (F, variable < domain_end),
            (after_domain_val, True),
        )

    expr_domain_pairs = F.as_expr_set_pairs()
    pieces = [(before_domain_val, variable < domain_start)]

    for (expr, interval) in expr_domain_pairs:
        if interval.end <= domain_start or interval.start >= domain_end:
            continue
        if abs(interval.end - domain_start) < VALIDATION_EPSILON:
            continue
        if abs(interval.start - domain_end) < VALIDATION_EPSILON:
            continue

        upper_bound = domain_end if interval.end > domain_end else interval.end
        pieces.append((expr, variable < upper_bound))

    pieces.append((after_domain_val, variable >= domain_end))
    return Piecewise(*pieces)


def subsequent_pairs(the_iterable):
    '''
    Given an iterable (a, b, c, d, ...) return a generator over
    pairs (a, b), (b, c), (c, d), ...

    Return an empty iterable if
    '''
    it = iter(the_iterable)

    try:
        pair_first = next(it)
        pair_second = next(it)
    except StopIteration:
        return

    while True:
        yield (pair_first, pair_second)
        try:
            pair_first = pair_second
            pair_second = next(it)
        except StopIteration:
            return
