import numpy as np
import placentagen as pg


def strahler_from_input(elems,orders,order_system,offset_order):
    split_order = orders[order_system]
    inlet_order = split_order[0]
    if order_system == 'generation':
        offset_order *= -1
    interest_order = inlet_order-offset_order
    interested_elems = []
    print('Order:',order_system)
    print('Order interest:',interest_order)
    print('Inlet Order:',inlet_order)
    for i in range(0, len(split_order)):
        if split_order[i] == interest_order:
            interested_elems.append(elems[i,:])

    print('Length:',len(interested_elems))
    return np.asarray(interested_elems)

def generate_circle_from_inlet(nodes, inlet_node_idx,  radius, n_points):
    '''Returns the n coordinates along a circle of radius r centered around the inlet node'''
    inlet_node_x = nodes[inlet_node_idx, 1]
    inlet_node_y = nodes[inlet_node_idx, 2]
    inlet_node_z = nodes[inlet_node_idx, 3]
    degrees_per_point = 360/n_points
    degrees = np.linspace(0,2*np.pi,n_points)
    coordinates = []
    for angle in degrees:
        x_coord = (np.cos(angle)*radius)+inlet_node_x
        y_coord = (np.sin(angle)*radius)+inlet_node_y
        add_array = np.asarray([x_coord,y_coord, 10])
        coordinates.append(add_array)
    return np.asarray(coordinates)

def calculate_closest_node(nodes, circle_nodes):
    node_loc = nodes[:,1:4]
    closest_nodes = []
    for circle_point in circle_nodes:
        distances = np.linalg.norm(node_loc-circle_point, axis=1)
        closest_index = np.argmin(distances)
        closest_nodes.append(nodes[closest_index,:])
    return np.asarray(closest_nodes)

def connected_elems(circle_nodes, elems, flow):
    closest_node_indices = circle_nodes[:, 0]

    mask = np.isin(elems[:,1],closest_node_indices) | np.isin(elems[:,2],closest_node_indices)
    circle_elems = elems[mask]
    circle_flow = flow[mask]
    return circle_elems, circle_flow

def get_distance_from_inlet(inlet_coords, nodes,elems):
    coords = nodes[:,1:4]
    distance_nodes = np.linalg.norm(coords-inlet_coords[1:4], axis = 1)
    # Step 1: Create a dictionary to map node IDs to their coordinates
    node_dict = {int(row[0]): row[1:] for row in nodes}

    # Step 2: Extract coordinates of node_1 and node_2 for each element
    node_1_coords = np.array([node_dict[node_id] for node_id in elems[:, 1]])
    node_2_coords = np.array([node_dict[node_id] for node_id in elems[:, 2]])

    # Step 3: Calculate midpoints of each element
    midpoints = (node_1_coords + node_2_coords) / 2

    # Step 4: Calculate distances from midpoints to the inlet
    distance_elems = np.linalg.norm(midpoints - inlet_coords[1:4], axis=1)
    return distance_nodes,distance_elems