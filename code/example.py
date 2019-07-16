from silent_duel import *

from sympy import Expr
from sympy import Lambda
from sympy import N
from sympy import Symbol


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
