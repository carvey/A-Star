from a_star import p1


def astar(start):
    """
    Perform A* search

    :param a_star.Node start: Start node
    :return:
    """
    # The set of currently discovered nodes still to be evaluated.
    # Initially, only the start node is known.
    open_set = set()
    open_set.add(start)

    # The set of nodes already evaluated.
    closed_set = set()

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, came_from will eventually contain the
    # most efficient previous step.
    came_from = {}  # the empty map

    # For each node, the cost of getting from the start node to that node.
    g_score = {node: float('inf') for node in nodes}  # map with default value of Infinity

    # # The cost of going from start to start is zero.
    g_score[start] = 0

    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    f_score = {node: float('inf') for node in nodes}  # map with default value of Infinity

    # For the first node, that value is completely heuristic.
    f_score[start] = heuristic_cost_estimate(start)

    while len(open_set) > 0:
        # Pick the node from the open set with the lowest f_score
        current = min(f_score[node] for node in open_set)

        # Move node from the open set to the closed set
        open_set.remove(current)
        closed_set.add(current)

        # Reached the goal state
        if current.goal_check():
            return reconstruct_path(came_from, current)

        for neighbor in current.neighbors():
            # Ignore neighbors that have already been evaluated
            if neighbor in closed_set:
                continue

            new_path_to_neighbor = g_score[current] + dist_between(current, neighbor)
            old_path_to_neighbor = g_score[neighbor]

            # Neighbor has not been evaluated yet
            if new_path_to_neighbor < old_path_to_neighbor or neighbor not in open_set:
                neighbor.f_cost = 123
                neighbor.parent = current

                # Update came_from dict
                came_from[neighbor] = current
                g_score[neighbor] = new_path_to_neighbor
                f_score[neighbor] = g_score[neighbor] + heuristic_cost_estimate(neighbor, goal)

                # Add neighbor to the open set if it doesn't already exist
                if neighbor not in open_set:
                    open_set.add(neighbor)

    # while len(open_set) > 0:
    #     current = min(node.f_score() for node in open_set)  # the node in open_set having the lowest f_score[] value
    #     if current == goal:
    #         return reconstruct_path(came_from, current)
    #
    #     open_set.remove(current)
    #     closed_set.add(current)
    #     for neighbor in current.neighbors():
    #         if neighbor in closed_set:
    #             continue  # Ignore the neighbor which is already evaluated.
    #
    #         # The distance from start to a neighbor
    #         tentative_g_score = g_score[current] + dist_between(current, neighbor)
    #         if neighbor not in open_set:  # Discover a new node
    #             open_set.add(neighbor)
    #         elif tentative_g_score >= g_score[neighbor]:
    #             continue  # This is not a better path.
    #
    #         # This path is the best until now. Record it!
    #         came_from[neighbor] = current
    #         g_score[neighbor] = tentative_g_score
    #         f_score[neighbor] = g_score[neighbor] + heuristic_cost_estimate(neighbor, goal)
    #
    # return False


def dist_between(node1, node2):
    return 0


def heuristic_cost_estimate(node1):
    return 0


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path


nodes = [node for node in p1.map.values()]

print(astar(nodes[0]))
