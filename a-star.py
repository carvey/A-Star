

class Node:

    def __init__(self, num):
        """
        :param num: Node number
        """
        self.num = num

    def __str__(self):
        return self.num

    def __repr__(self):
        return self.__str__()

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
        pos = 1
        for val in file.read():
            if val.strip("\n").isalnum():
                node = Node(val)
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
        for pos, val in self.map.items():
            if pos % 3 == 0:
                print(val)
            else:
                print(val, end=" ")



p1 = Puzzle('sample-problems/p1', True)
p1.print_puzzle()