"""
A* Implementation of the 8 puzzle problem
Charles Arvey
Michael Palmer

Goal State:
0 1 2
3 4 5
6 7 8
"""


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
        :param val: Node number
        """
        self.val = val # actual value
        self.pos = pos # puzzle location -> where is this value on the board?

        self.start_pos = pos

        self.g = None
        self.h = None
        self.f = None

    def __str__(self):
        return "Node: %s" % self.val

    def __repr__(self):
        return self.__str__()

    def calc(self, g=True, h=False):
        """
        Heuristic will be the manhatten distance

        Can calculate the both the g and h costs with the associated flags
        :return:
        """
        start = None
        end = None

        if g:
            start = self.start_pos
            end = self.val

        if h:
            start = self.val
            end = self.pos

        start_coords = coord_map[self.val]
        start_x = start_coords[0]
        start_y = start_coords[1]

        goal_coords = coord_map[self.pos]
        goal_x = goal_coords[0]
        goal_y = goal_coords[1]

        dst = abs(start_x - goal_x) + abs(start_y - goal_y)

        return dst

    def calc_f(self):
        return self.calc(g=True) + self.calc(h=True)

    def validate_goal_state(self):
        """
        Will make sure that value of the node aligns with its position

        In this case, the optimal puzzle value will have the empty space at pos 0
        :return:
        """
        return self.val == self.pos

    def valid_movement_positions(self, position):

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
        self.map = None

        if file:
            self.map = self.parse_file(data)

        else:
            self.map = self.parse_puzzle_string()

    def parse_file(self, file):
        file = open(file)
        node_map = {}
        pos = 0
        for val in file.read():
            if val.strip("\n").isalnum():
                node = Node(int(val), pos)
                node_map[pos] = node
                pos += 1


        return node_map

    def parse_puzzle_string(self):
        """
        Read puzzle data and create a node tree from a string
        :return:
        """
        pass

    def print_puzzle(self):
        cnt = 1

        for pos, node in self.map.items():
            if cnt % 3 == 0:
                print(node.val)
            else:
                print(node.val, end=" ")

            cnt +=1

    def get_start_node(self):
        for pos, node in self.map.items():
            if node.val == 0:
                return node


    def neighbors(self, node):
        valid_movement_positions = node.valid_movement_positions(node.pos)
        neighbor_nodes = {pos:_node for pos,_node in self.map.items() if pos in valid_movement_positions}

        return neighbor_nodes

    def move_node(self, moving_node, empty_node):
        """
        Switches a real node with the node holding the val 0
        :param moving_node: The node that is being moved into the "empty" space, which is just a node with val 0
        :param empty_node:
        :return:
        """
        if empty_node.val != 0:
            raise Exception("Can't move to a position that does not contain the 0 value")

        # create dummy vars to hold the positions of each node while we switch
        moving_pos = moving_node.pos
        empty_pos = empty_node.pos

        # switch the nodes in Puzzles position, val map
        self.map[empty_pos] = moving_node
        self.map[moving_pos] = empty_node

        # switch the position on the nodes themselves
        self.map[empty_pos].pos = moving_pos
        self.map[moving_pos].pos = empty_pos


    def solve(self):
        current_node = self.get_start_node()

        open = [current_node]
        closed = []

        # to speed this up in the future, we can keep a list of nodes that have reached goal state and check length
        while True:
            neighbors = self.neighbors(current_node)
            neighbor_values = {}

            for pos, neighbor in neighbors.items():
                neighbor_values[neighbor.calc_f()] = neighbor

            lowest_f_node = neighbor_values[min(neighbor_values)]
            current_node = neighbors[lowest_f_node.pos]

            open.pop(current_node)
            closed.append(current_node)

            # is path found ??
            if self.validate_goal_state():
                return



    def validate_goal_state(self):
        for pos, node in self.map.items():
            if not node.validate_goal_state():
                return False

        return False



p1 = Puzzle('sample-problems/p1', True)
p1.print_puzzle()
p1.solve()