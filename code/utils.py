from sympy.solvers import solve
from sympy import Piecewise


def solve_unique_real(expr, var, solution_min=0, solution_max=1, allow_no_solutions=False):
    '''
    Solve an equation and return any solution in the given range. This should
    be used when the caller knows that there is a unique solution to the
    equation in the given range.

    If the range bounds are set to None, then the assumption is that there is a
    unique global real solution. By default the range bound is [0,1].
    '''
    expr = expr.simplify()
    if isinstance(expr, Piecewise):
        '''sympy often fails to solve equations that are piecewise, even
        if the other pieces are trivial like constants or zero.'''
        pieces = [x[0] for x in expr.args if not x[0].is_constant()]
        sol_sets = [
            solve_unique_real(
                expr=e,
                var=var,
                solution_min=solution_min,
                solution_max=solution_max,
                allow_no_solutions=True)
            for e in pieces
        ]
        solutions = [sol for sol_set in sol_sets for sol in sol_set]
    else:
        solutions = solve(expr, var, minimal=True, quick=True)
        solutions = [x for x in solutions if x.is_real]

    if solution_min is not None:
        solutions = [x for x in solutions if x >= solution_min]
    if solution_max is not None:
        solutions = [x for x in solutions if x <= solution_max]

    if allow_no_solutions:
        assert len(solutions) in [0, 1], "Expected unique solution but found {}".format(solutions)
        return [float(x) for x in solutions]
    else:
        assert len(solutions) == 1, "Expected unique solution but found {}".format(solutions)
        return float(solutions[0])


def mask_piecewise(F, variable, domain_start, domain_end, before_domain_val=0, after_domain_val=0):
    '''
    Given a piecewise, add conditions (0, variable < domain_start)
    and (0, variable > domain_end)

    Assumes the piecewise given as input has conditions specified
    only by their upper bound. (very specific to this project)
    '''
    F = F.simplify()
    if not isinstance(F, Piecewise):
        return Piecewise(
            (before_domain_val, variable <= domain_start),
            (after_domain_val, variable >= domain_end),
            (F, True)
        )
    expr_domain_pairs = F.as_expr_set_pairs()
    pieces = [(before_domain_val, variable < domain_start)]

    for (expr, interval) in expr_domain_pairs:
        if interval.end <= domain_start or interval.start >= domain_end:
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
