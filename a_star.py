"""
A* Implementation of the 8 puzzle problem
Charles Arvey
Michael Palmer
"""

import copy
import random
import cProfile
import timeit
import time
import argparse

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

# define and collect user arguments
parser = argparse.ArgumentParser("Specify a file containing a sample problem.")
parser.add_argument("--file", type=str, required=True, help="puzzle file to parse")
options = parser.parse_args()


class Node:

    def __init__(self, val, pos):
        """
        :param int val: Node number
        :param int pos: Position
        """
        self.val = val  # actual value
        # self.pos = pos # puzzle location -> where is this value on the board?

        # self.start_pos = pos

    def __str__(self):
        return "Node: %s" % self.val

    def __repr__(self):
        return self.__str__()

    @property
    def display_value(self):
        if self.val == 0:
            return " "

        else:
            return self.val

    def valid_movement_positions(self, position):
        """
        'A little janky'

        :param int position: Janky position
        :return: Janky Value
        :rtype: tuple
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


class Puzzle:
    """
    To represent one puzzle instance based on a sample file
    """

    def __init__(self, data, file=False):
        """

        :param data: Can be string 0 - 8 or file path (if file path, set flag)
        :param file: Flag to set if passing in file
        :return:
        """
        start_state = None
        goal_state = None

        if file:  # the data being passed in is a file to be parsed
            start_state, goal_state = self.parse_file(data)

        else:  # the data being passed in is a string containing a start and goal state
            start_state, goal_state = self.parse_full_data_string(data)

        # self.state = PuzzleState(state)
        self.start_state = PuzzleState(start_state)
        self.goal_state = PuzzleState(goal_state)

    def parse_file(self, file):
        """
        Parse the file of the format:

        7 3 8
        0 2 5
        1 4 6

        0 1 2
        3 4 5
        6 7 8

        Where the first puzzle state is the initial state, and the second state is the goal state

        :param str file: Path to the file
        :return: Parsed file
        :rtype: tuple
        """
        # Open the file
        file = open(file)

        # get the entire contents of the file
        data = file.read()

        # parse both the start and initial state
        return self.parse_full_data_string(data)

    def parse_data(self, data):
        """

        :param str data: string containing just one string state -> start state or goal state
        :return:
        :rtype: dict
        """
        pos = 0
        state_map = {}

        data = data.replace(" ", "").replace("\n", "")
        for val in data:
            # node = Node(int(val), pos)
            state_map[pos] = int(val)
            pos += 1

        return state_map

    def parse_full_data_string(self, full_data_string):
        """
        :param str full_data_string: a string containing the start state and goal state,
                                        delimited with two newline chars
        :return: A tuple of the format (starting state dictionary, goal state dictionary)
        :rtype: tuple
        """
        data = full_data_string.split("\n\n")

        start_data = data[0]
        goal_data = data[1]

        start_map = self.parse_data(start_data)
        goal_map = self.parse_data(goal_data)

        return start_map, goal_map

    def expand(self, state):
        """
        Take in a state, expand the puzzle state from the empty node, create a new puzzle state, and add the new
        state to the graph
        :param PuzzleState state: a PuzzleState instance to be expanded
        :return:
        """
        empty_node = state.get_empty_node()
        f_costs = {}

        # Print the current state
        print(state.print_state())

        for pos, child in state.children(empty_node).items():
            # Make a deep copy of the current state
            copied_state = copy.deepcopy(state)

            # Move the node in the new copy
            copied_state.move_node(child, empty_node)

            # Create a new puzzle state with the move reflected
            new_state = PuzzleState(copied_state.state)

            # Check if we have reached the goal state
            if new_state.validate_goal_state():
                print("Done:\n %s" % new_state.print_state())
                return

            empty_node = new_state.get_empty_node()

            # Calculate the aggregate f costs
            f_costs[new_state] = new_state.aggregate_f_costs

        # Find the minimum f cost
        min_value = f_costs[min(f_costs, key=f_costs.get)]

        # Find all the states with that minimum f cost
        min_states = [_state for _state, f in f_costs.items() if f == min_value]

        # Pick a random state
        next_state = random.choice(min_states)

        next_state.parent = state

        # Expand the next state
        self.expand(next_state)

    @staticmethod
    def random_min(iterable, key=lambda x: x):
        """
        Pick a random min using the key

        :param set iterable: Set of items
        :param key: Lambda expression to use to find the value of each item
        :return: Minimum value
        :rtype: PuzzleState
        """
        # Convert set to list
        items = list(iterable)

        # Shuffle it
        random.shuffle(items)

        # Find the min
        return min(items, key=key)

    @staticmethod
    def state_in(item, sequence):
        """
        Check if this state is in a sequence

        :param item:
        :param set sequence:
        :return: Boolean
        :rtype: bool
        """
        for x in sequence:
            if x.state == item.state:
                return True
        return False

    def solve2(self):
        """
        Solve it!
        :return: Solution
        :rtype: PuzzleState
        """
        start_state = self.start_state
        goal_state = self.goal_state
        open = set()
        closed = set()
        open.add(start_state)

        # moves = 0

        while len(open) > 0:
            current = self.random_min(open, key=lambda item: item.aggregate_f_costs)
            # print('Moves: %d' % moves)
            # print(current.print_state())

            open.remove(current)
            closed.add(current)

            if current.state == goal_state.state:
                return current

            cost = current.aggregate_f_costs

            for child in current.actions():
                if self.state_in(child, closed):
                    continue

                if child.aggregate_f_costs < cost or not self.state_in(child, open):
                    # current.f = current.aggregate_f_costs
                    child.parent = current

                    if not self.state_in(child, open):
                        open.add(child)

            # moves += 1

    def solve(self):
        """
        You know, do janky solving things that don't actually work.
        :return:
        """
        start_state = self.start_state  # a puzzle state instance

        if start_state.validate_goal_state():
            # solution found - do something here
            return

        # path = []
        self.expand(start_state)

    @staticmethod
    def print_path(state):
        global moves
        if state is None:
            moves = 0
            return
        Puzzle.print_path(state.parent)
        print('Move #%d' % moves)
        print(state.print_state())
        moves += 1

    def run_stats(self, run_times=5):
        timer = timeit.Timer(stmt=self.solve2)
        times = timer.repeat(run_times, 1)
        avg = sum(times) / len(times)
        fails = [fail for fail in times if fail > 5]
        success_rate = 100 - (len(fails) / run_times * 100)

        print("Avg time over %s iterations: %s" % (run_times, avg))
        print("Success Rate: %s%%" % success_rate)
        print("Failure Count (iterations exceeding 5s): %s" % len(fails))
        print("Failures: %s" % fails)


class PuzzleState:

    # g = 0
    # h = float('inf')
    # f = float('inf')

    def __init__(self, state, parent=None):
        """

        :param dict state:
        :param parent:
        """

        self.state = state
        self.parent = parent
        # self.positions = {v: k for k, v in state.items()}

    def __str__(self):
        return self.print_state()

    def __repr__(self):
        return self.print_state()

    def valid_movement_positions(self, position):
        """
        'A little janky'

        :param int position: Janky position
        :return: Janky Value
        :rtype: tuple
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
        :return: True if all nodes are in their proper place, False otherwise
        :rtype: bool
        """
        for pos, node in self.state.items():
            if not self.validate_node_goal_position(node):
                return False

        return True

    def get_empty_node(self):
        """
        Get the node containing the empty value (0)

        :return: Start node
        :rtype: Node
        """
        for pos, node in self.state.items():
            if node == 0:
                return node

    def move_node(self, moving_node, empty_node):
        """
        Switches a real node with the node holding the val 0
        :param Node moving_node: The node that is being moved into the "empty" space, which is just a node with val 0
        :param Node empty_node:
        :rtype: None
        """
        if empty_node != 0:
            raise Exception("Can't move to a position that does not contain the 0 value")

        # create dummy vars to hold the positions of each node while we switch
        moving_pos = self.node_position(moving_node)
        empty_pos = self.node_position(empty_node)

        # switch the nodes in the puzzle states dict
        self.state[empty_pos] = moving_node
        self.state[moving_pos] = empty_node

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

        :param Node node: the node to search for
        :return: the position of the given node in the state
        :rtype: int
        """

        # This is 2-4x slower than the loop below. Why Michael????
        # return self.positions[node]

        for pos, _node in self.state.items():
            if _node == node:
                return pos

    def children(self, node):
        """
        Return the nodes that can be switched with a given node
        :param Node node: The node to find the valid switches for
        :return: dict of the format {node_position: node, node_position: node}
        :rtype: dict
        """
        node_pos = self.node_position(node)
        valid_movement_positions = self.valid_movement_positions(node_pos)
        children_nodes = {pos: _node for pos, _node in self.state.items() if pos in valid_movement_positions}

        return children_nodes

    def actions(self):
        """
        Generate the possible actions (PuzzleStates) that can be achieved from the current state

        :return: List of actions
        :rtype: list of PuzzleState
        """
        node = self.get_empty_node()
        node_pos = self.node_position(node)
        valid_movement_positions = self.valid_movement_positions(node_pos)
        actions = []
        for pos, child in self.state.items():
            if pos in valid_movement_positions:
                # Make a copy
                copied_state = copy.copy(self.state)
                new_state = PuzzleState(copied_state)

                # Move the node in the new copy
                new_state.move_node(child, node)

                # Add to actions
                actions.append(new_state)

        return actions

    def calc(self, node, g=False, h=False):
        """
        Heuristic will be the manhatten distance

        Can calculate the both the g and h costs with the associated flags
        :param Node node:
        :param bool g:
        :param bool h:
        :return:
        :rtype: int
        """

        current_node_position = self.node_position(node)

        start = None
        end = None

        if (g and h) or (not g and not h):
            raise Exception('A single heuristic must be specified')

        if g:
            start = current_node_position
            end = node

        elif h:
            start = current_node_position
            end = node

        start_coords = coord_map[start]
        start_x = start_coords[0]
        start_y = start_coords[1]

        goal_coords = coord_map[end]
        goal_x = goal_coords[0]
        goal_y = goal_coords[1]

        dst = abs(start_x - goal_x) + abs(start_y - goal_y)

        # if g:
        #     node.g = dst
        # elif h:
        #     node.h = dst

        return dst

    def calc_f(self, node):
        """
        Returns the sum of the g and h values for this node
        :return:
        :rtype: int
        """
        return self.calc(node, g=True) + self.calc(node, h=True)

    def validate_node_goal_position(self, node):
        """
        Will make sure that value of the node aligns with its position

        This will have to change once we have dynamic goal states, as a nodes value will not always line up with its
        goal position

        In this case, the optimal puzzle value will have the empty space at pos 0
        :return: True if in the goal state, otherwise False.
        :rtype: bool
        """
        pos = self.node_position(node)
        return node == pos

    # TODO: Probably doesn't make it any faster as a property, but this really needs to be a static value because
    # I bet this running multiple times in the loop is adding a lot of runtime.
    @property
    def aggregate_f_costs(self):
        """
        Get the cumulative f cost for an entire puzzle state. This is to give us an estimate on whether the move
        we are making will be getting us closer to our goal state or not.
        :return: the aggregate f cost for an entire puzzle state
        :rtype: int
        """
        f_cost = 0
        # loop over the state and add up each nodes f cost
        for pos, node in self.state.items():
            f = self.calc_f(node)
            f_cost += f

        return f_cost


start = time.time()
puzzle = Puzzle(options.file, True)
solution = puzzle.solve2()

print('Solution found in %s seconds, tracing back path to start node...' % (time.time() - start))
Puzzle.print_path(solution)

end = time.time()
total_run_time = end - start
print("Total Time elapsed: %s" % total_run_time)


print("---------")
# Comment these out as necessary
puzzle.run_stats(25)
# cProfile.run("puzzle.solve2()", sort="tottime")