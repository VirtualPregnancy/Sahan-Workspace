import numpy as np
import placentagen as pg
import matplotlib.image as mpimg
from skimage import filters, measure, color
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from skimage.morphology import skeletonize  #Compute the skeleton of a binary image
from skan import csr
from skan import Skeleton, summarize
from skan import draw

def read_png(filename, extract_colour):
    #This function reads in a png file and extract the relevant colour from the image
    img1 = mpimg.imread(filename)
    if extract_colour == 'all':
        img2 = img1
    elif extract_colour == 'r':
        img2 = img1[:, :, 0]
    elif extract_colour == 'g':
        img2 = img1[:, :, 1]
    elif extract_colour == 'b':
        img2 = img1[:, :, 2]
    else:  #default to all channels
        img2 = img1
    return img2


def generate_placenta_outline(image, pixel_spacing, thickness, outputfilename, debug_img, debug_file):
    edges = filters.sobel(image)
    binary_edges = edges > filters.threshold_otsu(edges)
    contours = measure.find_contours(binary_edges, level=0.8)

    # Assume the largest contour is the desired shape
    largest_contour = max(contours, key=len)

    # Plot the contour on the original image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(largest_contour[:, 1], largest_contour[:, 0], linewidth=2, color='red')
    ax.set_title('Detected Contour')
    if debug_img:
        plt.show()

    # Extract contour points
    contour_points_mm = [(x * pixel_spacing, y * pixel_spacing) for y, x in largest_contour]

    nodes = np.zeros((len(contour_points_mm) * 3, 4))
    node_count = 0
    for point in contour_points_mm:
        for dim in range(0, 3):
            nodes[node_count, 0] = node_count  #node number
            nodes[node_count, 1] = point[1]  # X coordinate
            nodes[node_count, 2] = point[0]  # Y coordinate
            if dim == 0:
                nodes[node_count, 3] = 0.0  # Y coordinate
            elif dim == 1:
                nodes[node_count, 3] = thickness / 2.0
            elif dim == 2:
                nodes[node_count, 3] = thickness
            node_count += 1
    if debug_file:
        pg.export_ex_coords(nodes[:, :][0:node_count + 1], 'placenta_3d', outputfilename, 'exnode')
        print('Exported Outline Data to: ', outputfilename)
    print('3D Placenta outline generation complete. File output: ', debug_file)
    return nodes[:, :][0:node_count + 1]


def fit_ellipse_2d(img, weight):
    #x coordinates of placental edge
    surface_points_x = np.nonzero(img)[0].astype(float)  # np.zeros([np.count_nonzero(img),2])

    #y coordinates of placental edge
    surface_points_y = np.nonzero(img)[1].astype(float)

    com_start = [np.mean(surface_points_x), np.mean(surface_points_y)]
    #rough centre point X
    x_radius_start = (np.max(surface_points_x) - np.min(surface_points_x)) / 2.
    #rough centre coordinate Y
    y_radius_start = (np.max(surface_points_y) - np.min(surface_points_y)) / 2.
    alpha_start = 0.
    #optimizes using least squared an ellipse that fits the outline of the placenta
    opt = least_squares(distance_from_ellipse,
                        [x_radius_start, y_radius_start, alpha_start, com_start[0], com_start[1]],
                        args=(surface_points_x, surface_points_y, weight), xtol=1e-8, verbose=0)

    return surface_points_x, surface_points_y, opt.x


def distance_from_ellipse(params, surface_x, surface_y, penalisation_factor):
    x_rad = params[0]
    y_rad = params[1]
    alpha = params[2]
    #offset surface to current COM
    surface_x = surface_x - params[3]
    surface_y = surface_y - params[4]

    A = ((np.cos(alpha) / x_rad) ** 2. + (np.sin(alpha) / y_rad) ** 2.) * np.multiply(surface_x, surface_x)
    B = 2.0 * np.cos(alpha) * np.sin(alpha) * (1. / x_rad ** 2. - 1. / y_rad ** 2.) * np.multiply(surface_x, surface_y)
    C = (np.sin(alpha) / x_rad) ** 2. + (np.cos(alpha) / y_rad) ** 2. * np.multiply(surface_x, surface_x)
    distance = A + B + C - 1.
    if (x_rad > np.max(abs(surface_x)) and y_rad > np.max(abs(surface_y))):
        distance = distance
    else:  #penalise the ellipsoid being inside the structure
        distance = distance * penalisation_factor

    distance = np.sum(distance ** 2)
    return distance


def generate_ellipse_hull(datapoints):
    mean = np.mean(datapoints[:, 2])
    max_x = 0
    min_x = 0

    index = 0
    thickness = max(datapoints[:, 2]) - min(datapoints[:, 2])
    rz = thickness / 2.0
    filtered_y_list = np.unique(datapoints[:, 1])
    maxmin_x = np.zeros((len(filtered_y_list), 5))
    #find bounds of x for each y slice
    for y_index in filtered_y_list:
        for point in datapoints:
            if point[1] == y_index:
                if point[0] < min_x:
                    min_x = point[0]
                if point[0] > max_x:
                    max_x = point[0]

                maxmin_x[index, 0] = y_index
                maxmin_x[index, 1] = min_x
                maxmin_x[index, 2] = max_x
        index += 1
        min_x = 0
        max_x = 0
    #Generate array of rx and a (offset) for each y slice

    for i in range(0, len(maxmin_x)):
        rx = (maxmin_x[i, 2] - maxmin_x[i, 1]) / 2.0
        a = (maxmin_x[i, 2] + maxmin_x[i, 1]) / 2.0
        maxmin_x[i, 3] = rx
        maxmin_x[i, 4] = a
    datapoints_ellipse = []
    for point in datapoints:
        indices = np.where(maxmin_x[:, 0] == point[1])[0]
        coord_check = check_in_ellipse(point[0], point[2], maxmin_x[indices, 3], maxmin_x[indices, 4], rz)
        if coord_check == True:
            datapoints_ellipse.append(point)

    return datapoints_ellipse, maxmin_x


def check_in_ellipse(x, z, rx, a, rz):
    in_ellipse = False
    check_value = (((x - a) / rx) ** 2) + ((z / rz) ** 2)
    if check_value < 1.0:
        in_ellipse = True
    return in_ellipse


def skeletonise_2d(img):
    #convert img to binary
    binary = img > 1.0e-6  #all non zeros
    sk = skeletonize(binary, method='zhang')  # skeletonize binary image
    return sk


def new_branch(mydegrees, branch_data, coordinates, pixel_graph, i, node_kount, elem_kount, nodes, elems, node_map):
    parent_list = np.zeros(3)
    #print('parent node in',i, mydegrees[i])
    #i is the 'old' node number, from the skeletonisation, indexed ftom 1

    continuing = True
    xcord, ycord = coordinates

    currentdegrees = 2  #dummy to enter loop
    while currentdegrees == 2:  #while a continuation branch
        count_new_vessel = 0  #not a new vessel
        for j in range(pixel_graph.indptr[i], pixel_graph.indptr[i + 1]):  #looking at all branches connected to inew
            inew = pixel_graph.indices[j]  #index of connected branch (old) indexed from one
            np_old = np.where(node_map == i)  #node number
            #need to find the index of
            if inew not in node_map:
                currentdegrees = mydegrees[inew]  #how many branches this node is connected to
                count_new_vessel = count_new_vessel + 1  #create a new vessel segment
                #Create new node
                node_kount = node_kount + 1  #create a new node
                node_map[node_kount] = inew  #Mapping old to new node number
                nodes[node_kount, 0] = node_kount  #new node number
                nodes[node_kount, 1] = xcord[inew]  #coordinates indexed to 'old' node number
                nodes[node_kount, 2] = ycord[inew]
                #plt.plot(coordinates[inew,0],coordinates[inew,1],'+')
                nodes[node_kount, 3] = 0.0  #dummy z-coord
                #Create new element
                elem_kount = elem_kount + 1
                elems[elem_kount, 0] = elem_kount
                elems[elem_kount, 1] = np_old[0]
                elems[elem_kount, 2] = node_kount
                iold = i
                i = inew
            if j == (pixel_graph.indptr[
                         i + 1] - 1) and count_new_vessel == 0:  # a get out which basically throws out loops
                mydegrees[i] = 10
                currentdegrees = 10

    if currentdegrees == 1:  #Terminal
        continuing = False
    elif currentdegrees == 3:  #bifurcation
        loops = False
        #need to check if isolated loop
        k = np.where(np.asarray(branch_data['node-id-src']) == i)
        if k[0].size >= 1:
            for i_branch in range(0, k[0].size):
                if branch_data['branch-type'][k[0][i_branch]] == 3:
                    loops = True
        if not loops:
            pl_kount = 0
            node_kount_bif = node_kount
            for j in range(pixel_graph.indptr[i], pixel_graph.indptr[i + 1]):
                inew = pixel_graph.indices[j]
                if inew not in node_map:
                    node_kount = node_kount + 1
                    node_map[node_kount] = inew
                    nodes[node_kount, 0] = node_kount
                    nodes[node_kount, 1] = xcord[inew]
                    nodes[node_kount, 2] = ycord[inew]
                    nodes[node_kount, 3] = 0.0  # dummy z-coord
                    elem_kount = elem_kount + 1
                    elems[elem_kount, 0] = elem_kount
                    elems[elem_kount, 1] = node_kount_bif
                    elems[elem_kount, 2] = node_kount
                    parent_list[pl_kount] = inew
                    pl_kount = pl_kount + 1
            continuing = True
        else:
            continuing = False
    else:  # for now exclude morefications and treat as a terminal
        continuing = False
        if mydegrees[i] == 4 or mydegrees[i] == 5:
            i_trif = i
            node_map_temp = np.copy(node_map)
            node_kount_temp = node_kount
            parent_list_trif = np.zeros(4, dtype=int)
            true_branches = np.zeros(4, dtype=int)
            true_trifurcation = True
            pl_kount = 0
            true_branches_kount = 0
            for j in range(pixel_graph.indptr[i], pixel_graph.indptr[i + 1]):
                inew = pixel_graph.indices[j]
                if inew not in node_map_temp:
                    if mydegrees[inew] != 2:
                        true_trifurcation = False
                    else:
                        parent_list_trif[pl_kount] = inew
                        pl_kount = pl_kount + 1
            if true_trifurcation:
                for br in parent_list_trif:
                    br_length = 0.0
                    i = br
                    currentdegrees = mydegrees[i]
                    while currentdegrees == 2:  # while a continuation branch
                        count_new_vessel = 0  # not a new vessel
                        for j in range(pixel_graph.indptr[i],
                                       pixel_graph.indptr[i + 1]):  # looking at all branches connected to inew
                            inew = pixel_graph.indices[j]  # index of connected branch (old) indexed from one
                            # need to find the index of
                            if inew not in node_map_temp:
                                currentdegrees = mydegrees[inew]  # how many branches this node is connected to
                                count_new_vessel = count_new_vessel + 1  # create a new vessel segment
                                br_length = br_length + 1
                                node_kount_temp = node_kount_temp + 1  # create a new node
                                node_map_temp[node_kount_temp] = inew  # Mapping old to new node number
                                i = inew
                            if j == (pixel_graph.indptr[
                                         i + 1] - 1) and count_new_vessel == 0:  # a get out which basically throws out loops
                                currentdegrees = 10
                    if br_length > 0.0:
                        #True branches
                        true_branches[true_branches_kount] = br
                        true_branches_kount = true_branches_kount + 1
            if true_branches_kount == 0:
                continuing = False
            elif true_branches_kount == 1:
                #This is a single branch that should continue from here
                inew = true_branches[0]
                pl_kount = 0
                node_kount_bif = node_kount
                node_kount = node_kount + 1
                node_map[node_kount] = inew
                nodes[node_kount, 0] = node_kount
                nodes[node_kount, 1] = xcord[inew]
                nodes[node_kount, 2] = ycord[inew]
                nodes[node_kount, 3] = 0.0  # dummy z-coord
                elem_kount = elem_kount + 1
                elems[elem_kount, 0] = elem_kount
                elems[elem_kount, 1] = node_kount_bif
                elems[elem_kount, 2] = node_kount
                parent_list[pl_kount] = inew
                pl_kount = pl_kount + 1
                continuing = True
            elif true_branches_kount == 2:
                #This is a single branch that should continue from here
                pl_kount = 0
                node_kount_bif = node_kount
                for br in range(0, 2):
                    inew = true_branches[br]
                    node_kount = node_kount + 1
                    node_map[node_kount] = inew
                    nodes[node_kount, 0] = node_kount
                    nodes[node_kount, 1] = xcord[inew]
                    nodes[node_kount, 2] = ycord[inew]
                    nodes[node_kount, 3] = 0.0  # dummy z-coord
                    elem_kount = elem_kount + 1
                    elems[elem_kount, 0] = elem_kount
                    elems[elem_kount, 1] = node_kount_bif
                    elems[elem_kount, 2] = node_kount
                    parent_list[pl_kount] = inew
                    pl_kount = pl_kount + 1
                continuing = True
            else:
                continuing = True
                #TRUE TRIFURCATION
                pl_kount = 0
                node_kount_bif = node_kount
                #create a branch to the first point
                inew = true_branches[0]
                node_kount = node_kount + 1
                node_map[node_kount] = inew
                nodes[node_kount, 0] = node_kount
                nodes[node_kount, 1] = xcord[inew]
                nodes[node_kount, 2] = ycord[inew]
                nodes[node_kount, 3] = 0.0  # dummy z-coord
                elem_kount = elem_kount + 1
                elems[elem_kount, 0] = elem_kount
                elems[elem_kount, 1] = node_kount_bif
                elems[elem_kount, 2] = node_kount
                parent_list[pl_kount] = inew
                pl_kount = pl_kount + 1
                # Create a second branch to the ave coord of the parent and the other two branches
                x_coord = nodes[node_kount_bif, 1]
                y_coord = nodes[node_kount_bif, 2]
                for br in range(1, 3):
                    x_coord = x_coord + xcord[true_branches[br]]
                    y_coord = y_coord + ycord[true_branches[br]]
                x_coord = x_coord / 3.
                y_coord = y_coord / 3.
                node_kount = node_kount + 1
                nodes[node_kount, 0] = node_kount
                nodes[node_kount, 1] = x_coord
                nodes[node_kount, 2] = y_coord
                nodes[node_kount, 3] = 0.0  # dummy z-coord
                elem_kount = elem_kount + 1
                elems[elem_kount, 0] = elem_kount
                elems[elem_kount, 1] = node_kount_bif
                elems[elem_kount, 2] = node_kount
                node_kount_bif = node_kount
                for br in range(1, 3):
                    inew = true_branches[br]
                    node_kount = node_kount + 1
                    node_map[node_kount] = inew
                    nodes[node_kount, 0] = node_kount
                    nodes[node_kount, 1] = xcord[inew]
                    nodes[node_kount, 2] = ycord[inew]
                    nodes[node_kount, 3] = 0.0  # dummy z-coord
                    elem_kount = elem_kount + 1
                    elems[elem_kount, 0] = elem_kount
                    elems[elem_kount, 1] = node_kount_bif
                    elems[elem_kount, 2] = node_kount
                    parent_list[pl_kount] = inew
                    pl_kount = pl_kount + 1

    return node_kount, elem_kount, nodes, elems, continuing, node_map, parent_list

def tellme_figtitle(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()
def skel2graph(sk, outputfilename, debug_file):



    plt.clf()
    plt.imshow(sk)

    plt.setp(plt.gca(), autoscale_on=True)

    tellme_figtitle('Click on, or near to the inlets')

    plt.waitforbuttonpress()

    pts = plt.ginput(n=1, show_clicks=True, mouse_add=1)


    #pts = [[2349, 309], [2427.2, 3185.5]]

    # Im guessing these are coordinates of the umbilical artery insertion
    inlets = np.asarray(pts)
    print('Inlets: ', inlets, len(inlets))

    #converts skeleton to graphical structure
    pixel_graph, coordinates = csr.skeleton_to_csgraph(sk)
    xcord, ycord = coordinates
    len_coord = len(xcord)
    mydegrees = np.zeros(len_coord, dtype=int)
    ne = 0
    closest = np.ones((len(inlets), 2)) * 1000.
    count_isolated = 0
    count_terminal = 0
    count_bifurcations = 0
    count_morefications = 0
    count_continuing = 0
    for i in range(1, len(pixel_graph.indptr) - 1):
        np1 = i - 1  #index for node
        num_attached = (pixel_graph.indptr[i + 1] - pixel_graph.indptr[i])  #looking at how many attachments it has
        if (num_attached == 0):
            count_isolated = count_isolated + 1
            mydegrees[i] = 0
        elif (num_attached == 1):
            count_terminal = count_terminal + 1
            #potental inlet
            for j in range(0, len(inlets)):
                distance = np.sqrt((xcord[i] - inlets[j, 1]) ** 2. + (ycord[i] - inlets[j, 0]) ** 2.)
                if (distance < closest[j, 0]):
                    closest[j, 0] = distance
                    closest[j, 1] = np1
            mydegrees[i] = 1
        elif num_attached == 2:
            count_continuing = count_continuing + 1
            mydegrees[i] = 2
        elif num_attached == 3:
            count_bifurcations = count_bifurcations + 1
            mydegrees[i] = 3
        else:
            count_morefications = count_morefications + 1
            mydegrees[i] = num_attached

    print('closest points to inlet ' + str(closest))
    print('num isolated points ' + str(count_isolated))
    print('num terminals ' + str(count_terminal))
    print('num bifurcations ' + str(count_bifurcations))
    print('num morefiurcations ' + str(count_morefications))

    branch_data = summarize(Skeleton(sk))
    node_map = np.zeros(len(xcord))
    nodes = np.zeros((len(xcord), 4))
    elems = np.zeros((len(xcord), 3), dtype=int)
    elem_cnct = np.zeros((len(xcord), 3), dtype=int)
    node_kount = -1
    elem_kount = -1
    for ninlet in range(0, len(inlets)):
        i = int(closest[ninlet, 1]) + 1  #start at the inlet
        node_kount = node_kount + 1
        node_map[node_kount] = i  #old node number
        nodes[node_kount, 0] = node_kount  #new node number
        nodes[node_kount, 1] = xcord[i]  #indexing coordinates array at old node number i
        nodes[node_kount, 2] = ycord[i]
        nodes[node_kount, 3] = 0.0  #dummy z-coordinate
        new_gen_parents = []
        node_kount, elem_kount, nodes, elems, continuing, node_map, branch_parents = new_branch(mydegrees, branch_data,
                                                                                                coordinates,
                                                                                                pixel_graph, i,
                                                                                                node_kount, elem_kount,
                                                                                                nodes, elems, node_map)
        if continuing:
            new_gen_parents = np.append(new_gen_parents, branch_parents[branch_parents > 0])
        cur_gen_parents = new_gen_parents
        while len(cur_gen_parents) > 0:
            new_gen_parents = []
            for k in range(0, len(cur_gen_parents)):
                i = int(cur_gen_parents[k])
                node_kount, elem_kount, nodes, elems, continuing, node_map, branch_parents = new_branch(mydegrees,
                                                                                                        branch_data,
                                                                                                        coordinates,
                                                                                                        pixel_graph, i,
                                                                                                        node_kount,
                                                                                                        elem_kount,
                                                                                                        nodes, elems,
                                                                                                        node_map)
                if continuing:
                    new_gen_parents = np.append(new_gen_parents, branch_parents[branch_parents > 0])
            cur_gen_parents = new_gen_parents

    if debug_file:
        pg.export_exelem_1d(elems[:, :][0:elem_kount + 1], 'arteries', outputfilename)
        pg.export_ex_coords(nodes[:, :][0:node_kount + 1], 'arteries', outputfilename, 'exnode')

    return pixel_graph, coordinates, nodes[:, :][0:node_kount + 1], elems[:, :][0:elem_kount + 1]


def map_nodes_to_hull(nodes, params, thickness, outputfilename, debug_file):
    z_level = thickness / 2.0
    slice_coordinates = params[:, 0]
    for i in range(0, len(nodes)):
        difference = np.abs(slice_coordinates - nodes[i, 2])
        closest_slice_index = np.argmin(difference)
        slice_params = params[closest_slice_index, :]
        z_closest_ellipse = z_level * np.sqrt(1 - (((nodes[i, 1] - slice_params[4]) / slice_params[3]) ** 2))
        #radius_check = (((nodes[i, 0] - slice_params[4]) / slice_params[3]) ** 2) + ((nodes[i, 3] / z_level) ** 2)
        if z_closest_ellipse < nodes[i, 3]:
            nodes[i, 3] = z_closest_ellipse
        else:
            nodes[i, 3] = z_level
    if debug_file:
        pg.export_ex_coords(nodes, 'arteries', outputfilename, 'exnode')
        print('Arterial nodes mapped to shaped hull exported to: ', outputfilename)

    return nodes


def find_root_nodes(nodes, elems):
    node_upstream_end = []
    node_downstream_end = []
    root_nodes = []
    elem_cncty = pg.element_connectivity_1D(nodes[:, 1:4], elems)
    elem_up = elem_cncty['elem_up']
    elem_down = elem_cncty['elem_down']
    for i in range(0, len(elem_down)):
        if (elem_up[i, 0]) == 0:
            node_upstream_end.append(i)
        if (elem_down[i, 0] == 0):
            node_downstream_end.append(i)
    for elements in node_upstream_end:
        node_number = elems[elements, 1]
        root_nodes.append(nodes[node_number, :])

    return np.asarray(root_nodes), np.asarray(node_upstream_end)


def create_umb_anastomosis(nodes, elems, umb_length, output_name, debug_file, inlet_type):
    root_nodes, root_elems = find_root_nodes(nodes, elems)
    if len(root_nodes) == 2 and inlet_type == 'double':
        #calculate coordinates of midpoint for anastomosis
        x_point = (root_nodes[0, 1] + root_nodes[1, 1]) / 2
        y_point = (root_nodes[0, 2] + root_nodes[1, 2]) / 2
        z_midpoint = (root_nodes[0, 3] + root_nodes[1, 3]) / 2.0
        z_point = z_midpoint + umb_length
        node_index = len(nodes)

        #create and append new nodes to end of node file
        new_node_1 = np.asarray([1, x_point, y_point, z_point])
        new_node_2 = np.asarray([0, x_point, y_point, (z_point + umb_length)])
        new_node_1 = new_node_1.reshape(1, 4)
        new_node_2 = new_node_2.reshape(1, 4)
        nodes[:, 0] = nodes[:, 0].astype(int) + int(2)
        nodes_combined = np.vstack([new_node_2, new_node_1])
        nodes_new = np.vstack([nodes_combined, nodes])

        #create edit and append elements to match newly created nodes
        elems[:, 0] += 3
        elems[:, 1] += 2
        elems[:, 2] += 2
        root2anas_1 = np.asarray([1, 1, elems[root_elems[0], 1]])
        root2anas_2 = np.asarray([2, 1, elems[root_elems[1], 1]])
        anas2anas = np.asarray([0, 0, 1])
        root2anas_1 = root2anas_1.reshape(1, 3)
        root2anas_2 = root2anas_2.reshape(1, 3)
        anas2anas = anas2anas.reshape(1, 3)
        elems_new = np.vstack([anas2anas, root2anas_1, root2anas_2, elems])
        elems_new = elems_new.astype(int)
    elif len(root_nodes) == 1 and inlet_type == 'single':
        x_point = root_nodes[0, 1]
        y_point = root_nodes[0, 2]
        z_point = root_nodes[0, 3] + umb_length
        new_node = np.asarray([0, x_point, y_point, z_point])
        new_node = new_node.reshape(1, 4)
        nodes[:, 0] = nodes[:, 0].astype(int) + int(1)
        nodes_new = np.vstack([new_node, nodes])
        #create edit and append elements to match newly created nodes
        elems[:, 0] += 1
        elems[:, 1] += 1
        elems[:, 2] += 1
        anas2anas = np.asarray([0, 0, 1])
        anas2anas = anas2anas.reshape(1, 3)
        elems_new = np.vstack([anas2anas, elems])
        elems_new = elems_new.astype(int)

    if debug_file:
        pg.export_exelem_1d(elems_new, 'arteries', output_name)
        pg.export_ex_coords(nodes_new, 'arteries', output_name, 'exnode')
        print('Umbilical cord added. Nodes and elems mapped to shaped hull exported to: ', output_name)

    return nodes_new, elems_new


def get_scale(scalebar_size, image_array):

    #binary = img > 1.0e-6  #all non zeros

    line_pixels = np.where(image_array > 0.5)

    if len(line_pixels[0]) == 0:
        raise ValueError("No line detected in the image")

    # Extract the x-coordinates of the line
    x_coords = line_pixels[1]

    # Calculate the length of the line in pixels
    length_in_pixels = np.max(x_coords) - np.min(x_coords) + 1
    print('Length of bar in pixels: ', length_in_pixels)


    # Calculate scale in mm/pixel
    scale_mm_per_pixel = scalebar_size / length_in_pixels
    scale_mm_per_pixel = np.round(scale_mm_per_pixel,4)

    return scale_mm_per_pixel

