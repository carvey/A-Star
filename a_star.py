
coord_map = {
    '0': (0,0),
    '1': (1,0),
    '2': (2,0),
    '3': (0,-1),
    '4': (1,-1),
    '5': (2,-1),
    '6': (0,-2),
    '7': (1,-2),
    '8': (2,-2)
}

class Node:

    def __init__(self, val, pos):
        """
        :param val: Node number
        """
        self.val = val # actual value
        self.pos = pos # puzzle location -> where is this value on the board?

        self.g = None
        self.h = None
        self.f = None

    def __str__(self):
        return self.val

    def __repr__(self):
        return self.__str__()

    def calc_h(self):
        """
        Heuristic will be the manhatten distance
        :return:
        """
        start_coords = coord_map[self.val]
        start_x = start_coords[0]
        start_y = start_coords[1]

        goal_coords = coord_map[str(self.pos)]
        goal_x = goal_coords[0]
        goal_y = goal_coords[1]

        dst = abs(start_x - goal_x) + abs(start_y - goal_y)

        return dst

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
                node = Node(val, pos)
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

        for pos, val in self.map.items():
            if cnt % 3 == 0:
                print(val)
            else:
                print(val, end=" ")

            cnt +=1

    def get_start_node(self):
        for pos, val in self.map.items():
            if val == 0:
                return pos, val

    def solve(self):
        current_node = self.get_start_node()



p1 = Puzzle('sample-problems/p1', True)
p1.print_puzzle()
