from sympy.solvers import solve



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
