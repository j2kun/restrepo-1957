'''Tests for ActionDistribution'''
import math
from assertpy import assert_that
from silent_duel import *

from sympy import Lambda
from sympy import Symbol


x = Symbol('x')


def test_action_distribution_draw_no_point_mass_id_cdf():
    dist = ActionDistribution(
        support_start=0,
        support_end=1,
        cumulative_density_function=Lambda((x,), x),
        point_mass=0,
    )

    def rng():
        return 0.3

    assert_that(dist.draw(uniform_random_01=rng)).is_equal_to(0.3)

    def rng():
        return 0.8

    assert_that(dist.draw(uniform_random_01=rng)).is_equal_to(0.8)


def test_action_distribution_draw_no_point_mass_linear_cdf():
    dist = ActionDistribution(
        support_start=0.3,
        support_end=0.8,
        cumulative_density_function=Lambda((x,), 2 * (x - 0.3)),
        point_mass=0,
    )

    def rng():
        return 0

    assert_that(dist.draw(uniform_random_01=rng)).is_equal_to(0.3)

    def rng():
        return 1.0

    assert_that(dist.draw(uniform_random_01=rng)).is_equal_to(0.8)

    def rng():
        return 0.5

    assert_that(dist.draw(uniform_random_01=rng)).is_equal_to(0.55)


def test_action_distribution_draw_with_point_mass_id_cdf():
    dist = ActionDistribution(
        support_start=0,
        support_end=1,
        cumulative_density_function=Lambda((x,), 0.5 * x),
        point_mass=0.5,
    )

    def rng():
        return 0.3

    assert_that(dist.draw(uniform_random_01=rng)).is_equal_to(0.6)

    def rng():
        return 0.8

    assert_that(dist.draw(uniform_random_01=rng)).is_equal_to(1.0)


def test_action_distribution_draw_with_point_mass_complex_cdf():
    dist = ActionDistribution(
        support_start=0.2,
        support_end=0.8,
        # integral = 0.102
        cumulative_density_function=Lambda((x,), x**3),
        point_mass=1 - 0.102,
    )

    def rng():
        return 0.05

    assert_that(dist.draw(uniform_random_01=rng)).is_close_to(
        0.36840314986, 1e-6)

    def rng():
        return 0.103

    assert_that(dist.draw(uniform_random_01=rng)).is_equal_to(0.8)
