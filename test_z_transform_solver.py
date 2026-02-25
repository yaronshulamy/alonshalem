import unittest

import sympy as sp

from z_transform_solver import solve_question_system


class ZTransformSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a, self.z = sp.symbols("a z", real=True)
        self.n = sp.symbols("n", integer=True, nonnegative=True)
        self.solution = solve_question_system(a=self.a, z=self.z, n=self.n)

    def test_transfer_function(self) -> None:
        expected = self.z**2 / (2 * (self.z**2 - self.a**2))
        diff = sp.simplify(self.solution.transfer_function - expected)
        self.assertEqual(diff, 0)

    def test_poles(self) -> None:
        self.assertEqual(set(self.solution.poles), {self.a, -self.a})

    def test_stability_condition(self) -> None:
        self.assertEqual(self.solution.stability_condition, sp.Abs(self.a) < 1)


if __name__ == "__main__":
    unittest.main()
