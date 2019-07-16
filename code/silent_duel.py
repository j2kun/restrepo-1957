from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import Deque
from typing import List
from typing import NewType
import random

from sympy import Expr
from sympy import Integral
from sympy import Lambda
from sympy import N
from sympy import Piecewise
from sympy import Symbol
from sympy import diff
from sympy.solvers import solve

from binary_search import BinarySearchHint
from binary_search import binary_search
from utils import solve_unique_real
from utils import subsequent_pairs

SuccessFn = NewType('SuccessFn', Lambda)
EPSILON = 1e-8


def DEFAULT_RNG():
    return random.random()


@dataclass
class SilentDuelInput:
    '''Class containing the static input data to the silent duel problem.'''
    player_1_action_count: int
    player_2_action_count: int
    player_1_action_success: SuccessFn
    player_2_action_success: SuccessFn


@dataclass
class ActionDistribution:
    '''The interval on which this distribution occurs.'''
    support_start: float
    support_end: float

    '''
    The cumulative density function for the distribution.

    May be improper if point_mass > 0.
    '''
    cumulative_density_function: Lambda

    '''
    If nonzero, corresponds to an extra point mass at the support_end.
    Only used in the last action in the optimal strategy.
    '''
    point_mass: float = 0

    t = Symbol('t', nonnegative=True)

    def draw(self, uniform_random_01=DEFAULT_RNG):
        '''Return a random draw from this distribution.

        Args:
         - uniform_random_01: a callable that accepts zero arguments
           and returns a uniform random float between 0 and 1. Defaults
           to using python's standard random.random
        '''
        if self.support_start >= self.support_end:  # treat as a point mass
            return self.support_start

        uniform_random_number = uniform_random_01()

        if uniform_random_number > 1 - self.point_mass - EPSILON:
            return self.support_end

        return solve_unique_real(
            self.cumulative_density_function(self.t) - uniform_random_number,
            self.t,
            solution_min=self.support_start,
            solution_max=self.support_end,
        )

    def validate(self):
        '''Ensure the action distribution has probability summing to 1.'''
        df = diff(self.cumulative_density_function(self.t), self.t)
        total_prob_in_support = N(Integral(df, (self.t, self.support_start, self.support_end)).doit())
        print("Validating. prob_mass={} point_mass={}".format(total_prob_in_support, self.point_mass))
        assert abs(self.point_mass + total_prob_in_support - 1) < EPSILON

    def __str__(self):
        rounded_df = N(diff(self.cumulative_density_function(self.t), self.t), 2)
        s = '({:.3f}, {:.3f}): dF/dt = {}'.format(
            self.support_start,
            self.support_end,
            rounded_df
        )
        if self.point_mass > 0:
            s += '; Point mass of {:G} at {:.3f}'.format(
                self.point_mass, self.support_end)
        return s


@dataclass
class Strategy:
    '''
    A strategy is a list of action distribution functions, each of which
    describes the probability of taking an action on the interval of its
    support.
    '''
    action_distributions: List[ActionDistribution]

    def draw_game_strategy(self):
        '''Returns a draw for each action in order.'''
        return [d.draw() for d in self.action_distributions]

    def transition_times(self):
        '''Returns the ordered list of action transition times for this player.'''
        times = [self.action_distributions[0].support_start]
        times.extend([d.support_end for d in self.action_distributions])
        return times

    def __str__(self):
        return '\n'.join([str(x) for x in self.action_distributions])


@dataclass
class SilentDuelOutput:
    p1_strategy: Strategy
    p2_strategy: Strategy

    def __str__(self):
        return 'P1:\n{}\n\nP2:\n{}'.format(self.p1_strategy, self.p2_strategy)


@dataclass
class IntermediateState:
    '''Class containing the intermediate state of the silent
    duel construction algorithm. This class is mutated as the
    algorithm progresses through computing the transition times
    and normalization constants for each action distribution.'''

    '''
    A list of the transition times compute so far. This field
    maintains the invariant of being sorted. Thus, the first element
    in the list is a_{i + 1}, the most recently computed value of
    player 1's transition times, and the last element is a_{n + 1} = 1.
    This value is set on initializtion with `new`.
    '''
    player_1_transition_times: Deque[Expr]

    '''
    Same as player_1_transition_times, but for player 2 with b_j
    and b_m.
    '''
    player_2_transition_times: Deque[Expr]

    '''
    The values of h_i so far, the normalizing constants for the action
    probability distributions for player 1. Has the same sorting
    invariant as the transition time lists.
    '''
    player_1_normalizing_constants: Deque[Expr]

    '''
    Same as player_1_normalizing_constants, but for player 2,
    i.e., the k_j normalizing constants.
    '''
    player_2_normalizing_constants: Deque[Expr]

    @staticmethod
    def new():
        '''Create a new state object, and set a_{n+1} = 1, b_{m+1}=1.'''
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
    player_action_success should correspond to P, opponent_action_success
    to Q, and larger_transition_times to the b_j.

    If the inputs are switched appropriately, f^* is computed for player 2.
    '''
    P: SuccessFn = player_action_success
    Q: SuccessFn = opponent_action_success

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
    P: SuccessFn = duel_input.player_1_action_success
    Q: SuccessFn = duel_input.player_2_action_success
    t = Symbol('t0', nonnegative=True, real=True)
    a_i = Symbol('a_i', positive=True, real=True)
    b_j = Symbol('b_j', positive=True, real=True)

    p1_transitions = intermediate_state.player_1_transition_times
    p2_transitions = intermediate_state.player_2_transition_times

    # the left end of the transitions arrays contain the smallest
    # (latest computed) transition time for each player.
    # these are set to 1 for an empty intermediate state, i.e. for a_n, b_m
    a_i_plus_one = p1_transitions[0]
    b_j_plus_one = p2_transitions[0]
    computing_a_n = a_i_plus_one == 1
    computing_b_m = b_j_plus_one == 1

    p1_fstar_parameters = list(p2_transitions)[:-1]  # ignore b_{m+1} = 1
    p1_fstar = f_star(P, Q, t, p1_fstar_parameters)
    # the a_i part
    if computing_a_n:
        p1_integrand = ((1 + alpha) - (1 - alpha) * P(t)) * p1_fstar
        p1_integral_target = 2 * (1 - alpha)
    else:
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

    # the b_j part
    p2_fstar_parameters = list(p1_transitions)[:-1]  # ignore a_{n+1} = 1
    p2_fstar = f_star(Q, P, t, p2_fstar_parameters)
    if computing_b_m:
        p2_integrand = ((1 + beta) - (1 - beta) * P(t)) * p2_fstar
        p2_integral_target = 2 * (1 - beta)
    else:
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

    # the h_i part
    h_i_integral = Integral(p1_fstar, (t, a_i, a_i_plus_one))
    h_i_integrated = h_i_integral.doit()
    h_i_numerator = (1 - alpha) if computing_a_n else 1
    h_i = h_i_numerator / h_i_integrated

    # the k_j part
    k_j_integral = Integral(p2_fstar, (t, b_j, b_j_plus_one))
    k_j_integrated = k_j_integral.doit()
    k_j_numerator = (1 - beta) if computing_b_m else 1
    k_j = k_j_numerator / k_j_integrated

    return (a_i, b_j, h_i, k_j)


def compute_as_and_bs(duel_input: SilentDuelInput,
                      alpha: float = 0,
                      beta: float = 0) -> IntermediateState:
    '''
    Compute the a's and b's for the silent duel, given a fixed
    alpha and beta as input.
    '''
    t = Symbol('t0', nonnegative=True, real=True)

    p1_index = duel_input.player_1_action_count
    p2_index = duel_input.player_2_action_count
    intermediate_state = IntermediateState.new()

    while p1_index > 0 or p2_index > 0:
        # the larger of a_i, b_j is kept as a parameter, then the other will be repeated
        # in the next iteration; e.g., a_{i-1} and b_j (the latter using a_i in its f^*)
        (a_i, b_j, h_i, k_j) = compute_ai_and_bj(
            duel_input, intermediate_state, alpha=alpha, beta=beta
        )

        # there is one exception, if a_i == b_j, then the computation of f^* in the next
        # iteration (I believe) should not include the previously kept parameter. I.e.,
        # in the symmetric version, if a_n is kept and the next computation of b_m uses
        # the previous a_n, then it will produce the wrong value.
        #
        # I resolve this by keeping both parameters when a_i == b_j.
        if abs(a_i - b_j) < EPSILON and p1_index > 0 and p2_index > 0:
            intermediate_state.add_p1(float(a_i), float(h_i))
            intermediate_state.add_p2(float(b_j), float(k_j))
            p1_index -= 1
            p2_index -= 1
        elif (a_i > b_j and p1_index > 0) or p2_index == 0:
            intermediate_state.add_p1(float(a_i), float(h_i))
            p1_index -= 1
        elif (b_j > a_i and p2_index > 0) or p1_index == 0:
            intermediate_state.add_p2(float(b_j), float(k_j))
            p2_index -= 1

    print("a_1 = {:.3f} b_1 = {:.3f}".format(
        intermediate_state.player_1_transition_times[0],
        intermediate_state.player_2_transition_times[0],
    ))
    return intermediate_state


def compute_piecewise_action_density(
        action_start: float,
        action_end: float,
        opponent_transition_times: List[float],
        normalizing_constant: float,
        player_action_success: SuccessFn,
        opponent_action_success: SuccessFn,
        variable: Symbol):
    # The density function is a piecewise h_i * f^* with discontinuities
    # at the values of opponent_transition_times contained in the interval.
    # Thus, we construct a piecewise function whose pieces correspond to
    # all the b_j that split the interval (a_i, a_{i+1}).
    piecewise_components = [(0, variable < action_start), (0, action_end < variable)]

    def add_piece(expr, variable, piece_start, piece_end):
        piecewise_components.append((expr, variable < piece_end))

    # chop off the last entry, which is always 1
    opponent_transition_times = list(opponent_transition_times)[:-1]

    piece_start = action_start
    piece_end = action_end
    for b_j in opponent_transition_times:
        if action_start + EPSILON < b_j < action_end - EPSILON:
            # break into a new piece
            piece_end = b_j
            larger_transition_times = [x for x in opponent_transition_times if x >= b_j - EPSILON]
            piece_fstar = f_star(
                player_action_success,
                opponent_action_success,
                variable,
                larger_transition_times,
            )
            add_piece(normalizing_constant * piece_fstar, variable, piece_start, piece_end)
            piece_start = b_j
            piece_end = action_end

    # at the end, add the last piece, which may be the only piece
    larger_transition_times = [x for x in opponent_transition_times if x >= action_end - EPSILON]
    piece_fstar = f_star(
        player_action_success,
        opponent_action_success,
        variable,
        larger_transition_times,
    )

    add_piece(normalizing_constant * piece_fstar, variable, piece_start, piece_end)
    return Piecewise(*piecewise_components)


def compute_strategy(
        player_action_success: SuccessFn,
        player_transition_times: List[float],
        player_normalizing_constants: List[float],
        opponent_action_success: SuccessFn,
        opponent_transition_times: List[float],
        time_1_point_mass: float = 0) -> Strategy:
    '''
    Given the transition times for a player, compute the action cumulative
    density functions for the optimal strategy of the player.
    '''
    action_distributions = []

    pairs = subsequent_pairs(player_transition_times)
    for (i, (action_start, action_end)) in enumerate(pairs):
        normalizing_constant = player_normalizing_constants[i]
        x = Symbol('x', real=True)
        piecewise_action_density = compute_piecewise_action_density(
            action_start,
            action_end,
            opponent_transition_times,
            normalizing_constant,
            player_action_success,
            opponent_action_success,
            x,
        )

        t = Symbol('t', real=True)
        '''
        piece_action_density is a(possible piecewise - defined) probability
        density function, and we need to integrate it to compute the
        cumulative density function.

        Integrating the whole function in one go will cause Sympy to use
        strange groupings and Min, which then gets Heaviside involved and
        reduces readability (IMO). piecewise_integrate seems to avoid this.

        Note piecewise_integrate doesn't replace the endpoints of a piecewise
        function's pieces, so we apply subs(x, t) to the output to update the
        domain endpoints. This has no effect on the function because there are
        no x's left after integrating.
        '''
        integrated = piecewise_action_density.piecewise_integrate(
            (x, action_start, t)).subs(x, t)
        piece_cdf = Lambda((t,), integrated)

        action_distributions.append(ActionDistribution(
            support_start=action_start,
            support_end=action_end,
            cumulative_density_function=piece_cdf
        ))

    action_distributions[-1].point_mass = time_1_point_mass
    return Strategy(action_distributions=action_distributions)


def compute_player_strategies(silent_duel_input, intermediate_state, alpha, beta):
    p1_strategy = compute_strategy(
        player_action_success=silent_duel_input.player_1_action_success,
        player_transition_times=intermediate_state.player_1_transition_times,
        player_normalizing_constants=intermediate_state.player_1_normalizing_constants,
        opponent_action_success=silent_duel_input.player_2_action_success,
        opponent_transition_times=intermediate_state.player_2_transition_times,
        time_1_point_mass=alpha,
    )
    p2_strategy = compute_strategy(
        player_action_success=silent_duel_input.player_2_action_success,
        player_transition_times=intermediate_state.player_2_transition_times,
        player_normalizing_constants=intermediate_state.player_2_normalizing_constants,
        opponent_action_success=silent_duel_input.player_1_action_success,
        opponent_transition_times=intermediate_state.player_1_transition_times,
        time_1_point_mass=beta,
    )
    return SilentDuelOutput(p1_strategy=p1_strategy, p2_strategy=p2_strategy)


def optimal_strategies(silent_duel_input: SilentDuelInput) -> SilentDuelOutput:
    '''Compute an optimal pair of corresponding strategies for the silent duel problem.'''
    # First compute a's and b's, and check to see if a_1 == b_1, in which case quit.
    intermediate_state = compute_as_and_bs(silent_duel_input, alpha=0, beta=0)
    a1 = intermediate_state.player_1_transition_times[0]
    b1 = intermediate_state.player_2_transition_times[0]

    if abs(a1 - b1) < EPSILON:
        return compute_player_strategies(
            silent_duel_input, intermediate_state, alpha=0, beta=0,
        )

    # Otherwise, binary search for an alpha/beta
    searching_for_beta = b1 < a1
    if searching_for_beta:
        def test(beta_value):
            new_state = compute_as_and_bs(
                silent_duel_input, alpha=0, beta=beta_value
            )
            new_a1 = new_state.player_1_transition_times[0]
            new_b1 = new_state.player_2_transition_times[0]
            found = abs(new_a1 - new_b1) < EPSILON
            return BinarySearchHint(found=found, tooLow=new_b1 < new_a1)
    else:  # searching for alpha
        def test(alpha_value):
            new_state = compute_as_and_bs(
                silent_duel_input, alpha=alpha_value, beta=0
            )
            new_a1 = new_state.player_1_transition_times[0]
            new_b1 = new_state.player_2_transition_times[0]
            found = abs(new_a1 - new_b1) < EPSILON
            return BinarySearchHint(found=found, tooLow=new_a1 < new_b1)

    search_result = binary_search(
        test, param_min=0, param_max=1, callback=print
    )
    assert search_result.found

    # the optimal (alpha, beta) pair have product zero.
    final_alpha = 0 if searching_for_beta else search_result.value
    final_beta = search_result.value if searching_for_beta else 0

    intermediate_state = compute_as_and_bs(
        silent_duel_input, alpha=final_alpha, beta=final_beta
    )
    print(intermediate_state.player_1_transition_times)
    print(intermediate_state.player_2_transition_times)
    player_strategies = compute_player_strategies(
        silent_duel_input, intermediate_state, final_alpha, final_beta
    )

    '''
    # validate the output distributions for safety
    for ad in player_strategies.p1_strategy.action_distributions:
        ad.validate()

    for ad in player_strategies.p2_strategy.action_distributions:
        ad.validate()
    '''

    return player_strategies
