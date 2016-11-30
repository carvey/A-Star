"""
A* Implementation of the 8 puzzle problem
Charles Arvey
Michael Palmer

Implementation Architecture:
When the implementation is run, a Puzzle instance is created. This instance will initialize the puzzle's start and
goal states based on the input file. A* search is used to find the best path from the start state to the goal state.
For each possible action (move), a PuzzleState instance is created to represent the current state of the puzzle.
Dictionaries are used in our PuzzleState class to track states and map the values of the puzzle board to their
respective positions.

Heuristics:
We used a combination of the Manhattan Distance and Linear Conflict for our heuristic.

Requirements:
- Python 2.7

Input:
The input file will contain the initial and goal board configuration.
A sample can look like this:

    7 2 4
    5 0 6
    8 3 1

    0 1 2
    3 4 5
    6 7 8

The first block of numbers is the start state and the second block of numbers is the goal state.

Output:
If the goal configuration is reachable in a finite number of steps, all board configurations from initial to
goal and the minimum number of steps required to reach the goal are printed out. If the goal configuration is not
reachable in a finite number of steps, "No solution" is displayed.

Usage:
python a_star.py --file [path to input file]
"""

from Queue import PriorityQueue
from argparse import ArgumentParser
from heapq import heapify
from time import time
from timeit import Timer

# Map tile positions to coordinates for use by Manhattan Distance
coord_map = {
    0: (0, 0),
    1: (1, 0),
    2: (2, 0),
    3: (0, -1),
    4: (1, -1),
    5: (2, -1),
    6: (0, -2),
    7: (1, -2),
    8: (2, -2)
}


class UnsolvablePuzzleError(Exception):
    """Thrown when a Puzzle cannot be solved in a finite number of steps."""
    pass


class Puzzle:
    """
    To represent one puzzle instance based on a sample file
    """

    def __init__(self, data, is_file=False):
        """
        Initialize the Puzzle

        :param str data: Can be string 0 - 8 or file path (if file path, set flag)
        :param bool is_file: Flag to set if passing in file
        :raises: UnsolvablePuzzleError
        """
        if is_file:  # the data being passed in is a file to be parsed
            start_state, goal_state = self.parse_file(data)

        else:  # the data being passed in is a string containing a start and goal state
            start_state, goal_state = self.parse_full_data_string(data)

        # Initialize start and goal states
        self.start_state = PuzzleState(state=start_state, puzzle=self)
        self.goal_state = PuzzleState(state=goal_state, puzzle=self)

        # Calculate the aggregate heuristic cost using manhattan and linear conflict
        self.start_state.calc_aggregate_costs()

        # Check if the puzzle is not solvable
        if not self.solvable():
            raise UnsolvablePuzzleError("This Puzzle is not solvable.")

    @staticmethod
    def parse_file(filename):
        """
        Parse the file of the format:

        7 3 8
        0 2 5
        1 4 6

        0 1 2
        3 4 5
        6 7 8

        Where the first puzzle state is the initial state, and the second state is the goal state

        :param str filename: Path to the file
        :return: Parsed file
        :rtype: tuple
        """
        # Open the file
        f = open(filename)

        # get the entire contents of the file
        data = f.read()

        # parse both the start and initial state
        return Puzzle.parse_full_data_string(data)

    @staticmethod
    def parse_data(data):
        """
        Parse state data

        :param str data: string containing just one string state -> start state or goal state
        :return: State dictionary
        :rtype: dict
        """
        pos = 0
        state_map = {}

        data = data.replace(" ", "").replace("\n", "")
        for val in data:
            state_map[pos] = int(val)
            pos += 1

        return state_map

    @staticmethod
    def parse_full_data_string(full_data_string):
        """
        Parse the full data string from a file and create appropriate start and goal maps

        :param str full_data_string: a string containing the start state and goal state,
                                        delimited with two newline chars
        :return: A tuple of the format (starting state dictionary, goal state dictionary)
        :rtype: tuple of dict
        """
        start_data, goal_data = full_data_string.split("\n\n")

        start_map = Puzzle.parse_data(start_data)
        goal_map = Puzzle.parse_data(goal_data)

        return start_map, goal_map

    @staticmethod
    def state_in(item, sequence):
        """
        Check if this state is in a sequence

        :param PuzzleState item: PuzzleState instance
        :param list of PuzzleState sequence:
        :return: Boolean
        :rtype: bool
        """
        return item.state in sequence

    def solve(self):
        """
        Solve it!

        :return: Solution
        :rtype: PuzzleState
        """
        open_states = PriorityQueue()
        closed_states = set()
        open_states_list = list()
        closed_states_list = list()

        # Add the start state to the frontier
        open_states.put((self.start_state.f, self.start_state.h, self.start_state.g, self.start_state))
        open_states_list.append(self.start_state.state)

        # Keep searching until the frontier is empty
        while open_states:

            # Pop from the priority queue and add the best state to the explored set
            current = open_states.get()[3]
            closed_states.add(current)

            # Update the actual states (dicts) that are in open/closed (this makes the search faster)
            open_states_list.remove(current.state)
            closed_states_list.append(current.state)

            # Have we reached the goal state?
            if current.validate_goal_state():
                return current

            # Loop through the possible actions from this state
            for child in current.actions():
                # If child is already in explored, skip to next child
                if self.state_in(child, closed_states_list):
                    continue

                # Set the child's parent
                child.parent = current

                # Add child to frontier if it's not in explored or frontier
                if not self.state_in(child, closed_states_list) or not self.state_in(child, open_states_list):
                    open_states.put((child.f, child.h, child.g, child))
                    open_states_list.append(child.state)

                # TODO: This whole block doesn't seem to be necessary at all with a priority queue...
                elif self.state_in(child, open_states_list) and current.f > child.f:
                    # found better path cost, so keeping child and removing current
                    open_states.queue.remove(current)
                    heapify(open_states.queue)
                    open_states.put((child.f, child.h, child.g, child))

                    # Update the open states list
                    open_states_list.remove(current.state)
                    open_states_list.append(child.state)

    def solvable(self):
        """
        Determine if this puzzle is solvable

        :return: True if the puzzle is solvable, otherwise False
        :rtype: bool
        """

        start_inversions = Puzzle.inversions(self.start_state.state)
        goal_inversions = Puzzle.inversions(self.goal_state.state)

        # the parity of inversions of the goal state and start state must be the same
        return (goal_inversions % 2 == 0) == (start_inversions % 2 == 0)

    @staticmethod
    def inversions(state):
        """
        There is a way to do this O(nlogn) instead of O(n^2) but can implement that later

        :param dict state: the mapping of positions to values
        :return: the number of inversions present in that state
        """
        inversions = 0
        values = list(state.values())
        values.remove(0)

        for i in range(7):
            for j in range(i+1, 8):
                if values[i] > values[j]:
                    inversions += 1
        return inversions

    @staticmethod
    def solution_path(state):
        """
        Trace the path back from the specified state to the start state

        :param PuzzleState state: Solution state
        :return: Solution path
        :rtype: list of PuzzleState
        """
        path = [state]

        while state.parent:
            state = state.parent
            path.append(state)

        return path

    @staticmethod
    def print_path(state):
        """
        Print the path from the start state to the specified state.

        :param PuzzleState state: Puzzle state instance
        :return: Number of moves
        :rtype: int
        """
        solution_path = Puzzle.solution_path(state)

        moves = 0
        # reversed just returns an iterator, so no lengthy operations being done on the list
        for sol in reversed(solution_path):
            print('Move #%d' % moves)
            print(sol.print_state())
            moves += 1
        return moves - 1

    def run_stats(self, run_times=5):
        """
        Run statistics

        :param int run_times: Number of times to run
        """
        timer = Timer(stmt=self.solve)
        times = timer.repeat(run_times, 1)
        avg = sum(times) / len(times)
        fails = [fail for fail in times if fail > 5]
        min_time = min(times)
        max_time = max(times)
        success_rate = 100 - (len(fails) / run_times * 100)

        print("Avg time over %s iterations: %s" % (run_times, avg))
        print("Minimum time: %s" % min_time)
        print("Maximum time: %s" % max_time)
        print("Success Rate: %s%%" % success_rate)
        print("Failure Count (iterations exceeding 5s): %s" % len(fails))
        print("Failures: %s" % fails)


class PuzzleState:

    def __init__(self, state=None, puzzle=None):
        """
        Initialize a puzzle state

        :param dict state: Puzzle state dictionary
        :param Puzzle puzzle: Puzzle instance
        """

        # Initialize g, h, and f cost values
        self.g = 0
        self.h = 0
        self.f = 0

        # Initialize parent to a null value
        self.parent = None

        # Pass in the state dict
        self.state = state

        # pass a reference the puzzle's goal state in order for this instance to check for a match
        self.puzzle = puzzle

    def __str__(self):
        """
        Return string representation of the puzzle state

        :return: Puzzle State as a string
        :rtype: str
        """
        return self.print_state()

    def __repr__(self):
        """
        Generate puzzle state representation

        :return: Puzzle State representation
        :rtype: str
        """
        return self.print_state()

    @staticmethod
    def valid_movement_positions(position):
        """
        Given the position of the empty square in the puzzle, determine the squares that can make a valid move.

        :param int position: Position
        :return: Valid movement positions
        :rtype: tuple of int
        """
        if position == 0:
            return 1, 3

        elif position == 1:
            return 0, 2, 4

        elif position == 2:
            return 1, 5

        elif position == 3:
            return 0, 4, 6

        elif position == 4:
            return 1, 3, 5, 7

        elif position == 5:
            return 2, 4, 8

        elif position == 6:
            return 3, 7

        elif position == 7:
            return 6, 4, 8

        elif position == 8:
            return 5, 7

    def validate_goal_state(self):
        """
        Check if the goal state has been reached

        :return: True if all nodes are in their proper place, False otherwise
        :rtype: bool
        """
        return self.state == self.puzzle.goal_state.state

    def move_node(self, moving_node):
        """
        Switches a real node with the node holding the val 0

        :param int moving_node: The node that is being moved into the "empty" space, which is just a node with val 0
        """
        # create dummy vars to hold the positions of each node while we switch
        moving_pos = self.node_position(moving_node)
        empty_pos = self.node_position(0)

        # switch the nodes in the puzzle states dict
        self.state[empty_pos] = moving_node
        self.state[moving_pos] = 0

        self.calc_aggregate_costs()

    def print_state(self):
        """
        Print the current state of the puzzle

        :return: a string representing the puzzles state
        :rtype: str
        """
        cnt = 1
        puzzle_state = ""
        for pos, node in self.state.items():
            if node == 0:
                node = " "
            if cnt % 3 == 0:
                puzzle_state += "%s\n" % str(node)
            else:
                puzzle_state += "%s " % str(node)

            cnt += 1

        return puzzle_state

    def node_position(self, node):
        """
        Returns the given nodes position in the current state

        :param int node: the node to search for
        :return: the position of the given node in the state
        :rtype: int
        """

        for pos, _node in self.state.items():
            if _node == node:
                return pos

    def actions(self):
        """
        Generate the possible actions (PuzzleStates) that can be achieved from the current state

        :return: List of actions
        :rtype: list of PuzzleState
        """
        node_pos = self.node_position(0)
        valid_movement_positions = PuzzleState.valid_movement_positions(node_pos)
        actions = []
        g_cost = self.g + 1

        for pos, child in self.state.items():
            if pos in valid_movement_positions:
                # Make a copy
                copied_state = self.state.copy()
                new_state = PuzzleState(state=copied_state, puzzle=self.puzzle)
                new_state.g = g_cost

                # Move the node in the new copy
                new_state.move_node(child)

                # Add to actions
                actions.append(new_state)

        return actions

    def calc_manhattan(self, pos, node):
        """
        Calculate the manhattan distance.

        Can calculate the both the g and h costs with the associated flags
        :param int pos: Current node position
        :param int node: Target node
        :return: Manhattan distance
        :rtype: int
        """
        end = self.puzzle.goal_state.node_position(node)
        current_x, current_y = coord_map[pos]
        goal_x, goal_y = coord_map[end]

        dst = abs(current_x - goal_x) + abs(current_y - goal_y)

        return dst

    def calc_linear_conflict(self):
        """
        Calculate the linear conflict of this state

        Two tiles tj and tk are in a linear conflict if:
         - tj and tk are in the same line
         - goal positions of tj and tk are both in that line
         - tj is to the right of tk
         - goal position of tj is to the left of the goal position of tk

        :return: Linear conflict
        :rtype: int
        """
        linear_vertical_conflict = 0
        linear_horizontal_conflict = 0

        rows = [
            [self.state[0], self.state[1], self.state[2]],
            [self.state[3], self.state[4], self.state[5]],
            [self.state[6], self.state[7], self.state[8]]
        ]

        # Calculate vertical conflicts
        for row, row_list in enumerate(rows):
            maximum = -1
            for col, value in enumerate(row_list):
                if value != 0 and (value - 1) / 3 == row:
                    if value > maximum:
                        maximum = value
                    else:
                        linear_vertical_conflict += 2

        cols = [
            [self.state[0], self.state[3], self.state[6]],
            [self.state[1], self.state[4], self.state[7]],
            [self.state[2], self.state[5], self.state[8]]
        ]

        # Calculate horizontal conflicts
        for col, col_list in enumerate(cols):
            maximum = -1
            for row, value in enumerate(col_list):
                if value != 0 and value % 3 == col + 1:
                    if value > maximum:
                        maximum = value
                    else:
                        linear_horizontal_conflict += 2

        return linear_vertical_conflict + linear_horizontal_conflict

    def calc_aggregate_costs(self):
        """
        Calculate the cumulative costs for an entire puzzle state. This is to give us an estimate on whether the move
        we are making will be getting us closer to our goal state or not.
        """
        self.h = 0

        # loop over the state and add up each nodes f, g, and h costs
        for pos, node in self.state.items():
            if node != 0:
                self.h += self.calc_manhattan(pos, node)

        self.h += self.calc_linear_conflict()
        self.f = self.g + self.h


if __name__ == "__main__":
    # define and collect user arguments
    parser = ArgumentParser("Specify a file containing a sample problem.")
    parser.add_argument("--file", type=str, required=True, help="puzzle file to parse")
    options = parser.parse_args()

    try:
        start_time = time()
        puzzle = Puzzle(options.file, True)
        solution = puzzle.solve()

        print('Solution found in %s seconds, tracing back path to start node...' % (time() - start_time))
        optimal_moves = Puzzle.print_path(solution)

        print('Optimal Moves: %d' % optimal_moves)

        end_time = time()
        total_run_time = end_time - start_time
        print("Total Time elapsed: %s seconds" % total_run_time)

        print("---------")

        # Comment these out as necessary
        # puzzle.run_stats(25)

    except UnsolvablePuzzleError:
        print('No solution - this puzzle is not solvable in a finite number of steps')
