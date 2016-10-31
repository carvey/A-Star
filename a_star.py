"""
A* Implementation of the 8 puzzle problem
Charles Arvey
Michael Palmer

Goal State:
0 1 2
3 4 5
6 7 8
"""
import matplotlib
import networkx as nx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import pylab

# # Initialize the graph
# graph = nx.DiGraph()
#
# # Add edges
# for edge in Main.__graph.edge_set:
#     graph.add_edge(edge.node_from.name, edge.node_to.name)
#
# # Set layout, draw the graph, and display it
# pos = nx.shell_layout(graph)
# nx.draw(graph, pos, node_size=1500, node_color='yellow', edge_color='red', with_labels=True)
# # Show the graph
# # pylab.show()


coord_map = {
    0: (0,0),
    1: (1,0),
    2: (2,0),
    3: (0,-1),
    4: (1,-1),
    5: (2,-1),
    6: (0,-2),
    7: (1,-2),
    8: (2,-2)
}


class Node:

    def __init__(self, val, pos):
        """
        :param int val: Node number
        :param int pos: Position
        """
        self.val = val # actual value
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

        :param int position:
        :return:
        """
        if position == 0:
            return (1, 3)

        elif position == 1:
            return (0, 2, 4)

        elif position == 2:
            return (1, 5)

        elif position == 3:
            return (0, 4, 6)

        elif position == 4:
            return (1, 3, 5, 7)

        elif position == 5:
            return (2, 4, 8)

        elif position == 6:
            return (3, 7)

        elif position == 7:
            return (6, 4, 8)

        elif position == 8:
            return (5, 7)


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

        if file: # the data being passed in is a file to be parsed
            start_state, goal_state = self.parse_file(data)

        else: # the data being passed in is a string containing a start and goal state
            start_state, goal_state = self.parse_full_data_string(data)

        # self.state = PuzzleState(state)
        self.start_state = PuzzleState(start_state)
        self.goal_state = PuzzleState(goal_state)

        self.graph = nx.DiGraph()

    def parse_data(self, data):
        """

        :param data: string containing just one string state -> start state or goal state
        :return:
        """
        pos = 0
        state_map = {}

        data = data.replace(" ", "").replace("\n", "")
        for val in data:
            node = Node(int(val), pos)
            state_map[pos] = node
            pos += 1

        return state_map

    def parse_file(self, file):
        """
        Parse the file

        :param file:
        :return:
        :rtype: dict
        """
        file = open(file)
        data = file.read() # get the entire contents of the file

        return self.parse_full_data_string(data) # parse both the start and initial state


    def parse_full_data_string(self, full_data_string):
        """
        :param full_data_string: a string containing the start state and goal state, deliminated with two newline chars
        :return:
        """
        data = full_data_string.split("\n\n")

        start_data = data[0]
        goal_data = data[1]

        start_map = self.parse_data(start_data)
        goal_map = self.parse_data(goal_data)

        return start_map, goal_map

    def expand(self):
        """
        Take in a state, expand the puzzle state from the empty node, create a new puzzle state, and add the new
        state to the graph
        :return:
        """
        pass

    def solve(self):
        """
        You know, do solving things that don't actually work.

        :return:
        """
        start_state = self.start_state
        self.graph.add_node(start_state)

        if start_state.validate_goal_state():
            # solution found - do something here
            return

        # find neighbors (expand current state, finding all possible node moves) and calculate f costs
        empty_node = start_state.get_empty_node()
        for pos, neighbor in start_state.neighbors(empty_node).items():
            copied_state = start_state
            copied_state.move_node(neighbor, empty_node)

            puzzle_state = PuzzleState(copied_state.state)

            self.graph.add_edge(start_state, puzzle_state)


        # open = set()
        # open.add(start_node)
        # closed = set()
        #
        # # to speed this up in the future, we can keep a list of nodes that have reached goal state and check length
        # while len(open) > 0:
        #     self.print_puzzle()
        #     current_node = min(open, key=lambda x: x.calc_f())
        #
        #     open.remove(current_node)
        #     closed.add(current_node)
        #
        #     if self.validate_goal_state():
        #         return
        #
        #     neighbors = self.neighbors(current_node)
        #
        #     for pos, neighbor in neighbors.items():
        #         if neighbor in closed:
        #             continue
        #
        #         if neighbor.calc_f() < current_node.calc_f() or neighbor not in open:
        #
        #             neighbor.parent = current_node
        #             if neighbor not in open:
        #                 open.add(neighbor)

        # neighbor_values = {}
        # for pos, neighbor in neighbors.items():
        #     neighbor_values[neighbor.calc_f()] = neighbor
        #
        # lowest_f_node = neighbor_values[min(neighbor_values)]
        # current_node = neighbors[lowest_f_node.pos]
        # is path found ??
        # if self.validate_goal_state():
        #     return

class PuzzleState:

    def __init__(self, state):

        self.state = state

    def __str__(self):
        return self.print_state()

    def __repr__(self):
        return self.print_state()

    def validate_goal_state(self):
        """
        Validate whether all nodes are in their proper place
        :return:
        :rtype: bool
        """
        for pos, node in self.state.items():
            if not node.validate_goal_state():
                return False

        return True

    def get_empty_node(self):
        """
        Get the starting node

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
        :return:
        """
        if empty_node.val != 0:
            raise Exception("Can't move to a position that does not contain the 0 value")

        # create dummy vars to hold the positions of each node while we switch
        moving_pos = moving_node.pos
        empty_pos = empty_node.pos

        # switch the nodes in Puzzles position, val map
        self.state[empty_pos] = moving_node
        self.state[moving_pos] = empty_node

        # # switch the position on the nodes themselves
        # self.state[empty_pos].pos = moving_pos
        # self.state[moving_pos].pos = empty_pos

    def print_state(self):
        """

        :param PuzzleState puzzle_state:
        :return:
        """
        cnt = 1
        puzzle_state = ""
        for pos, node in self.state.items():
            if cnt % 3 == 0:
                puzzle_state += "%s\n" % str(node.display_value)
            else:
                puzzle_state += "%s " % node.display_value

            cnt +=1

        return puzzle_state





    def node_position(self, node):
        """
        Returns the given nodes position in the current state
        :param node: the node to search for
        :return: the position of the given node in the state
        :rtype: int
        """
        for pos, _node in self.state.items():
            if _node == node:
                return True


    def neighbors(self, node):
        """
        Return the nodes that can be switched with a given node
        :param node: The node to find the valid switches for
        :return: dict of the format {node_position: node, node_position: node}
        :rtype: dict
        """
        valid_movement_positions = node.valid_movement_positions(node.pos)
        neighbor_nodes = {pos:_node for pos,_node in self.state.items() if pos in valid_movement_positions}

        return neighbor_nodes

    #TODO Need to verify this is working properly after the logic update
    def calc(self, node, g=True, h=False):
        """
        Heuristic will be the manhatten distance

        Can calculate the both the g and h costs with the associated flags
        :param bool g:
        :param bool h:
        :return:
        :rtype: int
        """

        start_node_pos = self.node_position(node)

        start = None
        end = None

        if g:
            start = node.start_pos
            end = node.val

        if h:
            start = node.val
            end = start_node_pos

        start_coords = coord_map[node.val]
        start_x = start_coords[0]
        start_y = start_coords[1]

        goal_coords = coord_map[node.pos]
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

        In this case, the optimal puzzle value will have the empty space at pos 0
        :return: True if in the goal state, otherwise False.
        :rtype: bool
        """
        pos = self.node_position(node)
        return node.val == pos


p1 = Puzzle('sample-problems/p1', True)
state = p1.start_state.print_state()
print(state)
p1.solve()

# nx.draw_circular(p1.graph, with_labels=True, node_size=3500, node_color='white')
# plt.show()