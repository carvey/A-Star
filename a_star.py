"""
A* Implementation of the 8 puzzle problem
Charles Arvey
Michael Palmer
"""

import copy
import random

import matplotlib
import networkx as nx

matplotlib.use('TkAgg')

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


class Node:

    def __init__(self, val, pos):
        """
        :param int val: Node number
        :param int pos: Position
        """
        self.val = val  # actual value
        # self.pos = pos # puzzle location -> where is this value on the board?

        self.start_pos = pos

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

        self.graph = nx.DiGraph()

    def show_graph(self):
        """
        Show the graph
        """
        pos = nx.shell_layout(self.graph)
        nx.draw(self.graph, pos, node_size=1500, node_color='yellow', edge_color='red', with_labels=True)

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
            node = Node(int(val), pos)
            state_map[pos] = node
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

            # Add node and edge to the graph
            self.graph.add_node(new_state)
            self.graph.add_edge(state, new_state)

            # Check if we have reached the goal state
            if new_state.validate_goal_state():
                print("Done:\n %s" % new_state.print_state())
                return

            empty_node = new_state.get_empty_node()

            # Calculate the aggregate f costs
            f_costs[new_state] = new_state.aggregate_f_costs()

        # Find the minimum f cost
        min_value = f_costs[min(f_costs, key=f_costs.get)]

        # Find all the nodes with that minimum f cost
        min_nodes = [_state for _state, f in f_costs.items() if f == min_value]

        # Pick a random node
        next_node = random.choice(min_nodes)

        # Expand the next node
        self.expand(next_node)

    def solve(self):
        """
        You know, do janky solving things that don't actually work.
        :return:
        """
        start_state = self.start_state  # a puzzle state instance
        self.graph.add_node(start_state)  # add the initial puzzle state to the graph

        if start_state.validate_goal_state():
            # solution found - do something here
            return

        # path = []
        self.expand(start_state)


class PuzzleState:

    def __init__(self, state):

        self.state = state

    def __str__(self):
        return self.print_state()

    def __repr__(self):
        return self.print_state()

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
            if node.val == 0:
                return node

    def move_node(self, moving_node, empty_node):
        """
        Switches a real node with the node holding the val 0
        :param Node moving_node: The node that is being moved into the "empty" space, which is just a node with val 0
        :param Node empty_node:
        :rtype: None
        """
        if empty_node.val != 0:
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
            if cnt % 3 == 0:
                puzzle_state += "%s\n" % str(node.display_value)
            else:
                puzzle_state += "%s " % node.display_value

            cnt += 1

        return puzzle_state

    def node_position(self, node):
        """
        Returns the given nodes position in the current state

        :param node: the node to search for
        :return: the position of the given node in the state
        :rtype: int
        """
        for pos, _node in self.state.items():
            if _node.val == node.val:
                return pos

    def children(self, node):
        """
        Return the nodes that can be switched with a given node
        :param node: The node to find the valid switches for
        :return: dict of the format {node_position: node, node_position: node}
        :rtype: dict
        """
        node_pos = self.node_position(node)
        valid_movement_positions = node.valid_movement_positions(node_pos)
        children_nodes = {pos: _node for pos, _node in self.state.items() if pos in valid_movement_positions}

        return children_nodes

    def calc(self, node, g=True, h=False):
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

        if g:
            start = current_node_position
            end = node.val

        elif h:
            start = current_node_position
            end = node.val

        start_coords = coord_map[start]
        start_x = start_coords[0]
        start_y = start_coords[1]

        goal_coords = coord_map[end]
        goal_x = goal_coords[0]
        goal_y = goal_coords[1]

        dst = abs(start_x - goal_x) + abs(start_y - goal_y)

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
        return node.val == pos

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


p1 = Puzzle('sample-problems/p1', True)
state = p1.start_state.print_state()
print(state)
p1.solve()

# nx.draw_circular(p1.graph, with_labels=True, node_size=3500, node_color='white')
# plt.show()
