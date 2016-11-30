import unittest
from a_star import Puzzle, UnsolvablePuzzleError


class PuzzleTests(unittest.TestCase):

    def test_inversions(self):
        """
        Test the the Puzzle staticmethod 'inversions'
        min inversions: 0
        max inversions: 36
        """
        # To represent a puzzle state with values 0, 1, 2, 3, 4, 5, 6, 7, 8 (inversions: 0)
        state1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
        state1_inversions = Puzzle.inversions(state1)

        self.assertEqual(state1_inversions, 0)

        # To represent a puzzle state with values 0, 1, 2, 7, 4, 5, 6, 3, 8 (inversions: 7)
        state2 = {0: 0, 1: 1, 2: 2, 3: 7, 4: 4, 5: 5, 6: 6, 7: 3, 8: 8}
        state2_inversions = Puzzle.inversions(state2)

        self.assertEqual(state2_inversions, 7)

        # To represent a puzzle state with values 8, 7, 6, 5, 4, 3, 2, 1, 0 (inversions: 28)
        state3 = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0}
        state3_inversions = Puzzle.inversions(state3)

        self.assertEqual(state3_inversions, 28)

    def test_solvable(self):
        # TODO Puzzle should accept more ways to input start/goal states, or solvable be static and take in state dicts
        # start inversions: 16, goal inversions: 0
        # 16 and 0 both even, so solvable should be true
        puzzle1_string = """
        7 3 8
        0 2 5
        1 4 6

        0 1 2
        3 4 5
        6 7 8
        """
        puzzle = Puzzle(puzzle1_string)
        solvable = puzzle.solvable()
        self.assertTrue(solvable)

        # start inversions: 16, goal inversions: 16
        # 16 and 16 are both even, so solvable should be true
        puzzle2_string = """
        7 3 8
        0 2 5
        1 4 6

        7 3 8
        0 2 5
        1 4 6
        """
        puzzle2 = Puzzle(puzzle2_string)
        solvable2 = puzzle2.solvable()
        self.assertTrue(solvable2)

        # start inversions: 16, goal inversions: 17
        # 16 is even but 17 is odd, so solvable should be false
        puzzle3_string = """
        7 3 8
        0 2 5
        1 4 6

        7 3 8
        0 2 5
        4 1 6
        """
        with self.assertRaises(UnsolvablePuzzleError):
            puzzle3 = Puzzle(puzzle3_string)
            self.assertFalse(puzzle3.solvable())


class SolveTests(unittest.TestCase):

    def test_solve_same_start_goal(self):
        puzzle1_string = """
        0 1 2
        3 4 5
        6 7 8

        0 1 2
        3 4 5
        6 7 8
        """
        puzzle1 = Puzzle(puzzle1_string)
        solution = puzzle1.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertEqual(len(solution_path), 1)

    def test_solve_dynamic_goal_state(self):
        puzzle_string = """
        1 2 3
        4 5 6
        7 0 8

        1 2 3
        4 5 6
        7 8 0
        """
        puzzle = Puzzle(puzzle_string)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertEqual(len(solution_path), 2)

    def test_test3_0_optimal_moves(self):
        puzzle = Puzzle('sample-problems/test3_0', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertEqual(len(solution_path) - 1, 28)

    def test_test3_5_optimal_moves(self):
        puzzle = Puzzle('sample-problems/test3_5', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertEqual(len(solution_path) - 1, 25)

    def test_test3_6_optimal_moves(self):
        puzzle = Puzzle('sample-problems/test3_6', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertEqual(len(solution_path) - 1, 11)

    def test_p4_optimal_moves(self):
        puzzle = Puzzle('sample-problems/p4', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertEqual(len(solution_path) - 1, 18)

    def test_p5_optimal_moves(self):
        puzzle = Puzzle('sample-problems/p5', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertEqual(len(solution_path) - 1, 22)

    def test_p6_optimal_moves(self):
        puzzle = Puzzle('sample-problems/p6', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertEqual(len(solution_path) - 1, 14)

    def test_solve_easy(self):
        puzzle = Puzzle('sample-problems/easy', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertGreater(len(solution_path) - 1, 1)

    def test_solve_medium(self):
        puzzle = Puzzle('sample-problems/medium', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertGreater(len(solution_path) - 1, 1)

    def test_solve_hard(self):
        puzzle = Puzzle('sample-problems/hard', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertGreater(len(solution_path) - 1, 1)

    def test_solve_worst(self):
        puzzle = Puzzle('sample-problems/worst', True)
        solution = puzzle.solve()
        solution_path = Puzzle.solution_path(solution)
        self.assertGreater(len(solution_path) - 1, 1)
