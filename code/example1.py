from silent_duel import *

from sympy import Lambda
from sympy import Symbol


x = Symbol('x')
P = Lambda((x,), x)
Q = Lambda((x,), x)

duel_input = SilentDuelInput(
    player_1_action_count=3,
    player_2_action_count=3,
    player_1_action_success=P,
    player_2_action_success=Q,
)

print("Input: {}".format(duel_input))
output = optimal_strategies(duel_input)
print(output)
output.validate()
