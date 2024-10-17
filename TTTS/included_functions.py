import numpy as np
import placentagen as pg
import matplotlib.image as mpimg
from skimage import filters, measure, color
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from skimage.morphology import skeletonize  #Compute the skeleton of a binary image
from skan import csr
from skan import Skeleton, summarize
from collections import Counter




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
def skel2graph(sk, outputfilename, debug_file, inlet_type):



    plt.clf()
    plt.imshow(sk)

    plt.setp(plt.gca(), autoscale_on=True)

    tellme_figtitle('Click on, or near to the inlets')

    plt.waitforbuttonpress()
    if inlet_type == 'double':
        pts = plt.ginput(n=2, show_clicks=True, mouse_add=1)
    elif inlet_type == 'single':
        pts = plt.ginput(n=1, show_clicks=True, mouse_add=1)
    elif inlet_type == 'TTTS':
        pts = plt.ginput(n=4, show_clicks=True, mouse_add=1)
    else:
        pts = plt.ginput(n=-1, show_clicks=True, mouse_add=1)

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
    plt.clf()
    plt.imshow(sk)
    tellme_figtitle('Click on, or near to the anastomoses')

    plt.waitforbuttonpress()
    anastomosis = plt.ginput(n=1, show_clicks=True, mouse_add=1)
    anast_node, anast_elem = find_closest_node(anastomosis[0][0], anastomosis[0][1], nodes[:, :][0:node_kount + 1],elems[:, :][0:elem_kount + 1])


    if debug_file:
        pg.export_exelem_1d(elems[:, :][0:elem_kount + 1], 'arteries', outputfilename)
        pg.export_ex_coords(nodes[:, :][0:node_kount + 1], 'arteries', outputfilename, 'exnode')

    return inlets, anast_node, anast_elem, nodes[:, :][0:node_kount + 1], elems[:, :][0:elem_kount + 1]

def find_closest_node(clicked_x, clicked_y, nodes, elems, endpoint=False):
    if not endpoint:
        distance = np.sqrt(((nodes[:,1]-clicked_x)**2) + ((nodes[:,2]-clicked_y)**2))
        closest_index = np.argmin(distance)
        closest_node = nodes[closest_index,:]
        mask = elems[:, 1] == closest_node[0]
        elem_closest = elems[mask]
    elif endpoint:
        elem_cncty = pg.element_connectivity_1D(nodes[:,1:4],elems)
        elem_up = elem_cncty['elem_up']
        elem_down = elem_cncty['elem_down']
        ending_node = []
        for i in range(0, len(elem_up)):
            if elem_down[i,0] == 0:
                ending_node.append(nodes[elems[i,2],:])
            elif elem_up[i,0] == 0:
               ending_node.append(nodes[elems[i,1],:])
        ending_node = np.asarray(ending_node)
        distance = np.sqrt(((ending_node[:,1]-clicked_x)**2) + ((ending_node[:,2]-clicked_y)**2))
        closest_index = np.argmin(distance)
        closest_node = ending_node[closest_index,:]
        mask = elems[:, 1] == closest_node[0]
        elem_closest = elems[mask]

    return closest_node, elem_closest


def find_branch_points(nodes, elems, anast_elem):
    elem_cnct = element_connectivity_multi(nodes[:,1:4],elems)
    elem_up = elem_cnct['elem_up']
    elem_down = elem_cnct['elem_down']
    biifurcation_elems = []
    root_node, root_elem = find_root_nodes(nodes,elems)
    bifurcation_nodes = []
    branch_parents = [root_elem[0]]
    terminals = pg.calc_terminal_branch(nodes[:,1:4], elems)
    terminal_elems = terminals['terminal_elems']
    if anast_elem != 0:
        mask = terminal_elems != anast_elem

    # Use the mask to filter the array
        terminal_elems = terminal_elems[mask]

    for el in range(0,len(elem_down)):
        if elem_down[el][0] == 2:
            biifurcation_elems.append(elems[el,:])
            bif_node = nodes[elems[el][2],:]
            bifurcation_nodes.append(bif_node)
            branch_parents.append(elem_down[el][1])
            branch_parents.append(elem_down[el][2])
    biifurcation_elems = np.asarray(biifurcation_elems)
    branch_end = np.hstack((biifurcation_elems[:,0],terminal_elems))
    return np.asarray(biifurcation_elems), np.asarray(bifurcation_nodes),np.asarray(branch_parents), np.sort(branch_end)

def allocate_branch_numbers(nodes, elems, anast_elem):
    bif_elems, bif_nodes, branch_parents, branch_end = find_branch_points(nodes,elems, anast_elem)
    elem_cnct = element_connectivity_multi(nodes[:,1:4],elems)
    elem_down = elem_cnct['elem_down']
    branch_data = []
    branch = 1
    elem_count = 0
    branch_start = branch_parents[branch - 1]
    branch_structure = np.zeros(len(elems))
    while branch <= len(branch_parents):
        if elem_down[elem_count,0] == 1: #element is continuation
            branch_structure[elem_count] = branch
            elem_count = elem_down[elem_count,1]
        elif elem_down[elem_count,0] == 2 or elem_down[elem_count,0] == 0: #Bifurcation or terminal
            branch_structure[elem_count] = branch
            branch_info = np.asarray([branch, branch_start, elem_count])
            branch_data.append(branch_info)
            if branch == len(branch_parents): #This will be the last point
                break
            branch += 1
            branch_start = branch_parents[branch - 1]
            elem_count = branch_start

    return branch_structure, np.asarray(branch_data)


def find_middle_index(branch_structure, branch_number):
    # Step 1: Extract indices corresponding to the branch number
    indices = np.where(branch_structure == branch_number)[0]

    # Step 2: Check if indices are found
    if len(indices) == 0:
        return None  # Or raise an exception, or return a default value

    # Step 3: Sort indices
    sorted_indices = np.sort(indices)

    # Step 4: Find the middle index
    middle_index = sorted_indices[len(sorted_indices) // 2]

    return middle_index
def allocate_stem_locations(branch_data, branch_structure ,terminals, anast_elem):
    stem_location_elems = []
    terminal_elem = terminals['terminal_elems']
    mask = terminal_elem != anast_elem

    # Use the mask to filter the array
    terminal_elem = terminal_elem[mask]
    for i in range(0, len(branch_data)):
        elem_start = branch_data[i,1]
        elem_end = branch_data[i,2]
        if elem_start != elem_end:
            if (elem_end not in terminal_elem):
                stem_end_elem = elem_end - 1 #is not terminal so add stem villi before to avoid trifurcation
            else:
                stem_end_elem = elem_end #is terminal so add stem villi at end
                stem_location_elems.append(stem_end_elem)
            middle_elem = find_middle_index(branch_structure,branch_data[i,0])
            if middle_elem != None:
                stem_location_elems.append(middle_elem)

        else: #Very short branch
            stem_location_elems.append(elem_end)
    return np.asarray(stem_location_elems)

def add_stem_villi(nodes_all, elems_all,sv_length,  terminals, anast_elem):
    branch_structure, branch_data = allocate_branch_numbers(nodes_all,elems_all, anast_elem)
    stem_location = allocate_stem_locations(branch_data,branch_structure, terminals, anast_elem)

    print("Number of stem villi added:", len(stem_location))
    node_len = len(nodes_all)
    elem_len = len(elems_all)
    new_node_len = node_len + len(stem_location)
    new_elem_len = elem_len + len(stem_location)

    #initialise new arrays
    chorion_elems = np.zeros((new_elem_len, 3), dtype=int)
    chorion_nodes = np.zeros((new_node_len, 4))
    node_count = node_len
    elem_count = elem_len
    chorion_nodes[0:node_len,:] = nodes_all
    chorion_elems[0:elem_len,:] = elems_all
    bif_elems, bif_nodes, branch_parents, branch_end = find_branch_points(nodes_all, elems_all, anast_elem)
    stem_location = np.unique(stem_location)
    for branch_elem in stem_location:
        connected_node_num = elems_all[branch_elem,2]
        connected_node = nodes_all[connected_node_num,:]
        if branch_elem not in bif_elems:
            chorion_nodes[node_count][0] = node_count #Node Number
            chorion_nodes[node_count][1] = connected_node[1]
            chorion_nodes[node_count][2] = connected_node[2]
            chorion_nodes[node_count][3] = connected_node[3] - sv_length
            chorion_elems[elem_count][0] = elem_count
            chorion_elems[elem_count][1] = connected_node_num
            chorion_elems[elem_count][2] = node_count
            elem_count += 1
            node_count += 1
    mask_nodes = np.any(chorion_nodes != 0, axis=1)
    mask_elems = np.any(chorion_elems != 0, axis=1)
    chorion_nodes = chorion_nodes[mask_nodes]
    chorion_elems = chorion_elems[mask_elems]

    return chorion_nodes, chorion_elems

def map_nodes_to_hull(nodes, params, thickness, outputfilename, debug_file):
    z_level = (thickness / 2.0)
    slice_coordinates = params[:, 0]
    for i in range(0, len(nodes)):
        difference = np.abs(slice_coordinates - nodes[i, 2])
        closest_slice_index = np.argmin(difference)
        slice_params = params[closest_slice_index, :]
        scale = 1

        z_closest_ellipse = z_level * np.sqrt(1 - (((nodes[i, 1] - slice_params[4]) / (slice_params[3]*scale)) ** 2))

        while(np.isnan(z_closest_ellipse)):
            scale += 0.01
            z_closest_ellipse = z_level * np.sqrt(1 - (((nodes[i, 1] - slice_params[4]) / (slice_params[3] * scale)) ** 2))

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

def find_parent_list(nodes, elems):
    node_downstream_end = []
    parent_nodes = []
    elem_cncty = element_connectivity_multi(nodes[:, 1:4], elems)
    elem_up = elem_cncty['elem_up']
    elem_down = elem_cncty['elem_down']
    for i in range(0, len(elem_down)):
        if (elem_down[i, 0] == 0):
            node_downstream_end.append(i)
    for elements in node_downstream_end:
        node_number = elems[elements, 1]
        parent_nodes.append(nodes[node_number, :])
    return np.asarray(parent_nodes), np.asarray(node_downstream_end)


def create_umb_anastomosis(nodes, elems, umb_length, output_name, debug_file, inlet_type, inlet_nodes,inlet_elems):
    if inlet_nodes is None:
        root_nodes, root_elems = find_root_nodes(nodes, elems)
    elif inlet_nodes is not None:
        root_nodes = []
        root_elems = []
        #for node in inlet_nodes:
            #root_nodes.append(nodes[int(node[0]),:])
            #rows_with_node = elems[elems[:, 1] == int(node[0])]
            #root_elems.append(np.asarray(rows_with_node[0,0]))
        root_nodes = inlet_nodes
        root_elems = inlet_elems
        pg.export_ex_coords(root_nodes, 'arteries', output_name + 'C', 'exnode')


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
    elif len(root_nodes) == 4 and inlet_type == 'TTTS':
        pairs = find_inlet_pairs(root_nodes)
        pair1_1 = pairs[0,0]
        pair1_2 = pairs[0,1]
        pair2_1 = pairs[1, 0]
        pair2_2 = pairs[1, 1]

        x_point = (root_nodes[pair1_1, 1] + root_nodes[pair1_2, 1]) / 2
        y_point = (root_nodes[pair1_1, 2] + root_nodes[pair1_2, 2]) / 2
        z_midpoint = (root_nodes[pair1_1, 3] + root_nodes[pair1_2, 3]) / 2.0
        z_point = z_midpoint + umb_length
        node_index = len(nodes)

        # create and append new nodes to end of node file
        new_node_1 = np.asarray([1, x_point, y_point, z_point])
        new_node_2 = np.asarray([0, x_point, y_point, (z_point + umb_length)])
        new_node_1 = new_node_1.reshape(1, 4)
        new_node_2 = new_node_2.reshape(1, 4)
        nodes[:, 0] = nodes[:, 0].astype(int) + int(2)
        nodes_combined = np.vstack([new_node_2, new_node_1])
        nodes_new = np.vstack([nodes_combined, nodes])

        # create edit and append elements to match newly created nodes
        elems[:, 0] += 3
        elems[:, 1] += 2
        elems[:, 2] += 2
        root2anas_1 = np.asarray([1, 1, elems[root_elems[pair1_1], 1]])
        root2anas_2 = np.asarray([2, 1, elems[root_elems[pair1_2], 1]])
        anas2anas = np.asarray([0, 0, 1])
        root2anas_1 = root2anas_1.reshape(1, 3)
        root2anas_2 = root2anas_2.reshape(1, 3)
        anas2anas = anas2anas.reshape(1, 3)
        elems_new = np.vstack([anas2anas, root2anas_1, root2anas_2, elems])
        elems_new = elems_new.astype(int)

        x_point = (root_nodes[pair1_1, 1] + root_nodes[pair1_2, 1]) / 2
        y_point = (root_nodes[pair1_1, 2] + root_nodes[pair1_2, 2]) / 2
        z_midpoint = (root_nodes[pair1_1, 3] + root_nodes[pair1_2, 3]) / 2.0
        z_point = z_midpoint + umb_length
        node_index = len(nodes)



        x_point = (root_nodes[pair2_1, 1] + root_nodes[pair2_2, 1]) / 2
        y_point = (root_nodes[pair2_1, 2] + root_nodes[pair2_2, 2]) / 2
        z_midpoint = (root_nodes[pair2_1, 3] + root_nodes[pair2_2, 3]) / 2.0
        z_point = z_midpoint + umb_length

        # create second inlet append new nodes to end of node file
        new_node_A = np.asarray([len(nodes_new), x_point, y_point, z_point])
        new_node_B = np.asarray([len(nodes_new)+1, x_point, y_point, (z_point + umb_length)])
        new_node_A = new_node_A.reshape(1, 4)
        new_node_B = new_node_B.reshape(1, 4)
        nodes_combined = np.vstack([new_node_A, new_node_B])
        nodes_new = np.vstack([nodes_new,nodes_combined])
        nodeA_num = new_node_A[0, 0]
        nodeB_num = new_node_B[0, 0]

        root2anas_1 = np.asarray([len(elems_new),nodeA_num,root_nodes[pair2_1,0]+2])
        root2anas_2 = np.asarray([len(elems_new)+1,nodeA_num,root_nodes[pair2_2,0]+2])
        anas2anas = np.asarray([len(elems_new)+2, nodeB_num, nodeA_num])
        root2anas_1 = root2anas_1.reshape(1, 3)
        root2anas_2 = root2anas_2.reshape(1, 3)
        anas2anas = anas2anas.reshape(1, 3)
        elems_new = np.vstack([elems_new,root2anas_1,root2anas_2,anas2anas])
        elems_new = elems_new.astype(int)
    else:
        elems_new = elems
        nodes_new = nodes


    if debug_file:
        pg.export_exelem_1d(elems_new, 'arteries', output_name)
        pg.export_ex_coords(nodes_new, 'arteries', output_name, 'exnode')
        print('Umbilical cord added. Nodes and elems mapped to shaped hull exported to: ', output_name)

    return nodes_new, elems_new



def find_inlet_pairs(root_nodes):
    pairs = np.array([[0,1],[2,3]])
    distance = 1000000
    paired_node = 1
    for i in range(1,4):
        distance_pair  = np.sqrt((root_nodes[i,1]-root_nodes[0,1])**2 + (root_nodes[i,2]-root_nodes[0,2])**2 + (root_nodes[i,3]-root_nodes[0,3])**2)
        if distance_pair < distance:
            distance = distance_pair
            paired_node = i
    if paired_node == 2:
        pairs = np.array([[0,2],[1,3]])
    elif paired_node == 3:
        pairs = np.array([[0,3],[1,2]])
    return pairs

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

def evaluate_orders(node_loc, elems):
    """Calculates generations, Horsfield orders, Strahler orders for a given tree
       Works for diverging trees only, but accounts for more than three elements joining at a node
       Inputs:
          node_loc = array with location of nodes
          elems = array with location of elements
    """
    num_elems = len(elems)
    # Calculate connectivity of elements
    elem_connect = element_connectivity_multi(node_loc, elems)
    elem_upstream = elem_connect['elem_up']
    elem_downstream = elem_connect['elem_down']
    # Initialise order definition arrays
    strahler = np.zeros(len(elems), dtype=int)
    horsfield = np.zeros(len(elems), dtype=int)
    generation = np.zeros(len(elems), dtype=int)

    # Calculate generation of each element
    maxgen = 1  # Maximum possible generation
    for ne in range(0, num_elems):
        ne0 = elem_upstream[ne][1]
        if elem_upstream[ne][0] != 0:
            # Calculate parent generation
            n_generation = generation[ne0]
            if elem_downstream[ne0][0] == 1:
                # Continuation of previous element
                generation[ne] = n_generation
            elif elem_downstream[ne0][0] >= 2:
                # Bifurcation (or morefurcation)
                if elem_downstream[ne][2] == elem_downstream[ne0][2] and elem_downstream[ne][2] != 0:
                    print('found anastomosis')
                generation[ne] = n_generation + 1
        else:
            generation[ne] = 1  # Inlet
        maxgen = np.maximum(maxgen, generation[ne])

    # Now need to loop backwards to do ordering systems

    for ne in range(num_elems - 1, -1, -1):

        n_horsfield = np.maximum(horsfield[ne], 1)
        n_children = elem_downstream[ne][0]
        if n_children == 1:
            if elem_downstream[elem_downstream[ne][1]][0] == 0:

                n_children = 0
        temp_strahler = 0
        strahler_add = 1
        if n_children >= 2:  # Bifurcation downstream
            temp_strahler = strahler[elem_downstream[ne][1]]  # first daughter
            for noelem in range(1, n_children + 1):
                ne2 = elem_downstream[ne][noelem]
                temp_horsfield = horsfield[ne2]
                if temp_horsfield > n_horsfield:
                    n_horsfield = temp_horsfield
                if strahler[ne2] < temp_strahler:
                    strahler_add = 0
                elif strahler[ne2] > temp_strahler:
                    strahler_add = 0
                    temp_strahler = strahler[ne2]  # strahler of highest daughter
            n_horsfield = n_horsfield + 1
        elif n_children == 1:
            ne2 = elem_downstream[ne][1]  # element no of daughter
            n_horsfield = horsfield[ne2]
            strahler_add = strahler[ne2]
        horsfield[ne] = n_horsfield
        strahler[ne] = temp_strahler + strahler_add

    return {'strahler': strahler, 'horsfield': horsfield, 'generation': generation}

def evaluate_orders_multi(nodes,elems, anast_elem):
    num_elems = len(elems)
    inlet_nodes_list, inlet_elems = find_root_nodes(nodes,elems)
    bif_elems, bif_nodes, branch_parents, branch_end = find_branch_points(nodes,elems, anast_elem)

    num_inlets = len(inlet_nodes_list)
    terminals = pg.calc_terminal_branch(nodes[:,1:4],elems)
    terminal_elems = terminals['terminal_elems']
    parent_nodes, parent_elems = find_parent_list(nodes, elems)
    # Calculate connectivity of elements
    # Initialise order definition arrays
    branching_path = []
    strahler = np.zeros(len(elems), dtype=int)
    horsfield = np.zeros(len(elems), dtype=int)
    generation = np.zeros(len(elems), dtype=int)
    ec = element_connectivity_multi(nodes[:,1:4],elems)
    elem_upstream = ec['elem_up']
    elem_down = ec['elem_down']
    inlet_for_parent = np.zeros(len(terminal_elems),dtype=int)
    for terminal in terminal_elems:
        if terminal == anast_elem:
            inlet, branch_path, = allocate_inlet(terminal,branch_parents,ec, 1)
            branching_path.append((terminal, branch_path[::-1]))
            inlet, branch_path, = allocate_inlet(terminal,branch_parents,ec, 2)
            branching_path.append((terminal, branch_path[::-1]))
        else:
            inlet, branch_path, = allocate_inlet(terminal,branch_parents,ec, 1)
            branching_path.append((terminal, branch_path[::-1]))
        #inlet_for_parent[i] = inlet

        #branching_path[terminal_elems[i]] = branch_path[::-1]
    #counter = Counter(inlet_for_parent)
    generation[inlet_elems[1]] = 1 #hardcode inlet2

    for parent, branching_elems in branching_path:
        branch_index = 0
        current_gen = 1
        elemcount = branching_elems[branch_index]
        while elemcount != parent:
            if elem_down[elemcount, 0] == 1:  # continuation of branch
                if generation[elemcount] == 0:
                    generation[elemcount] = current_gen
                elemcount = elem_down[elemcount, 1]  # move to downstream element
                if elemcount == parent:
                    if generation[elemcount] == 0:
                        generation[elemcount] = current_gen
            elif elem_down[elemcount,0] == 2:
                branch_index += 1
                if generation[elemcount] == 0:
                    generation[elemcount] = current_gen
                current_gen = generation[elemcount] +1
                elemcount = branching_elems[branch_index]
                if elemcount == parent:
                    if generation[elemcount] == 0:
                        generation[elemcount] = current_gen

    for terminal_elem in terminal_elems:
        strahler[terminal_elem]=1
        n_parents = elem_upstream[terminal_elem, 0]

        elemcount = elem_upstream[terminal_elem,1]

        inlet_found = False
        for i in range(1, n_parents + 1):

            while not inlet_found:
                if elemcount in inlet_elems:
                    inlet_found = True
                n_children =  elem_down[elemcount,0]
                if n_children == 1:
                    strahler[elemcount] = strahler[elem_down[elemcount,1]]
                if n_children == 2:
                    child1 = elem_down[elemcount, 1]
                    child2 = elem_down[elemcount, 2]
                    if strahler[child1] > strahler[child2]:
                        strahler[elemcount] = strahler[child1]
                    elif strahler[child1] < strahler[child2]:
                        strahler[elemcount] = strahler[child2]
                    elif strahler[child1] == strahler[child2]:
                        strahler[elemcount] = strahler[child2] + 1
                    #elemcount = elem_upstream[elemcount, 1]
                if elemcount not in inlet_elems:
                    elemcount = elem_upstream[elemcount, i]
                    n_parents = elem_upstream[elemcount, 0]

    #occurence = np.asarray([counter[inlet_elems[0]],counter[inlet_elems[1]]])
    branch_index = 0



    return {'strahler': strahler, 'horsfield': horsfield, 'generation': generation}

def allocate_inlet(parent_elem, branch_start, elem_connect, anast_path):
    inlet_elem = 0
    inlet_found = False

    next_elem = parent_elem
    elem_up = elem_connect['elem_up']
    branch_points = []
    while not inlet_found:
        if next_elem in branch_start:
            branch_points.append(next_elem)
        if elem_up[next_elem][0] == 1:
            next_elem = elem_up[next_elem][1]
        elif elem_up[next_elem][0] == 2:

            next_elem = elem_up[next_elem][anast_path]
        elif elem_up[next_elem][0] == 0:
            inlet_found = True
            inlet_elem = next_elem

            branch_points.append(inlet_elem)

    return inlet_elem, branch_points

def element_connectivity_multi(node_loc, elems):
    # Initialise connectivity arrays
    anastomosis = False
    num_elems = len(elems)
    num_nodes = len(node_loc)
    elems_at_node = np.zeros((num_nodes, 20), dtype=int) #allow up to 20-furcations
    # determine elements that are associated with each node
    for ne in range(0, num_elems):
        for nn in range(1, 3):
            nnod = elems[ne][nn]
            elems_at_node[nnod][0] = elems_at_node[nnod][0] + 1
            elems_at_node[nnod][elems_at_node[nnod][0]] = ne
    elem_upstream = np.zeros((num_elems, int(np.max(elems_at_node[:,0]))), dtype=int)
    elem_downstream = np.zeros((num_elems, int(np.max(elems_at_node[:,0]))), dtype=int)
    # assign connectivity
    for ne in range (0,num_elems):
        upstream_node = elems[ne][1]
        downstream_node = elems[ne][2]
        nnod2 = elems[ne][2]  # second node in elem
        if elems_at_node[nnod2][0] == 3:
            elem1 = elems_at_node[nnod2][1]
            elem2 = elems_at_node[nnod2][2]
            elem3 = elems_at_node[nnod2][3]
            if (elems[elem1, 2] == elems[elem2, 2]) or (elems[elem2, 2] == elems[elem3, 2]) or (
                    elems[elem1, 2] == elems[elem3, 2]):
                anastomosis = True
        for noelem in range(1, elems_at_node[nnod2][0] + 1):
            ne2 = elems_at_node[nnod2][noelem]

            if ne2 != ne:
                if not anastomosis:
                    elem_upstream[ne2][0] = elem_upstream[ne2][0] + 1
                    elem_upstream[ne2][elem_upstream[ne2][0]] = ne
                    elem_downstream[ne][0] = elem_downstream[ne][0] + 1
                    elem_downstream[ne][elem_downstream[ne][0]] = ne2
                elif anastomosis:
                    anastomosis = False
                    if elems[ne][2] != elems[ne2][2]:
                        elem_upstream[ne2][0] = elem_upstream[ne2][0] + 1
                        elem_upstream[ne2][elem_upstream[ne2][0]] = ne
                        elem_downstream[ne][0] = elem_downstream[ne][0] + 1
                        elem_downstream[ne][elem_downstream[ne][0]] = ne2
    return {'elem_up': elem_upstream, 'elem_down': elem_downstream}

def element_rearrage(nodes, elems, anast_elem):
    root_nodes, root_elems =  find_root_nodes(nodes, elems)
    bif_elems, bif_nodes, branch_parents, branch_end = find_branch_points(nodes,elems, anast_elem)
    terminals = pg.calc_terminal_branch(nodes[:,1:4],elems)
    terminal_elems = terminals['terminal_elems']
    branching_path = []

    num_elems  = len(elems)
    ec = element_connectivity_multi(nodes[:,1:4],elems)
    elem_upstream = ec['elem_up']
    elem_down = ec['elem_down']
    inlet_for_parent = np.zeros(len(terminal_elems),dtype=int)
    for terminal in terminal_elems:
        if terminal == anast_elem:
            inlet, branch_path, = allocate_inlet(terminal,branch_parents,ec, 1)
            branching_path.append((terminal, branch_path[::-1]))
            inlet, branch_path, = allocate_inlet(terminal,branch_parents,ec, 2)
            branching_path.append((terminal, branch_path[::-1]))
        else:
            inlet, branch_path, = allocate_inlet(terminal,branch_parents,ec, 1)
            branching_path.append((terminal, branch_path[::-1]))
    inlet1_elems = []
    inlet2_elems = []
    for i in range(0, num_elems):
        inlet, path = allocate_inlet(elems[i,0],branch_parents,ec,1)
        if inlet == 0:
            inlet1_elems.append(elems[i,:])
        else:
            inlet2_elems.append(elems[i,:])
    inlet1_elems = np.asarray(inlet1_elems)
    inlet2_elems = np.asarray(inlet2_elems)
    num_elems_branch1 = len(inlet1_elems)
    num_elems_branch2 = len(inlet2_elems)
    elemcount = root_elems[1]
    rearranged_branch = []
    rearranged_branch.append(elems[elemcount,0])
    for parent, branching_elems in branching_path:
        if branching_elems[0] != 0:
            branch_index = 0
            current_gen = 1
            elemcount = branching_elems[branch_index]
            while elemcount != parent:
                if elem_down[elemcount, 0] == 1:  # continuation of branch
                    if elemcount not in rearranged_branch:
                        rearranged_branch.append(elems[elemcount,0])
                    elemcount = elem_down[elemcount, 1]  # move to downstream element
                    if elemcount == parent:
                        if elemcount not in rearranged_branch:
                            rearranged_branch.append(elems[elemcount, 0])
                elif elem_down[elemcount,0] == 2:
                    branch_index += 1
                    if elemcount not in rearranged_branch:
                        rearranged_branch.append(elems[elemcount,0])

                    elemcount = branching_elems[branch_index]
                    if elemcount == parent:
                        if elemcount not in rearranged_branch:
                            rearranged_branch.append(elems[elemcount, 0])

    rearranged_branch.remove(anast_elem)
    rearranged_branch = np.asarray(rearranged_branch)
    elems_new = np.zeros((num_elems,3))
    for i in range (0,num_elems_branch1):
        elems_new[i,0] = i
        elems_new[i,1] = inlet1_elems[i,1]
        elems_new[i, 2] = inlet1_elems[i, 2]
    for i in range(0,len(rearranged_branch)):
        ne = i + num_elems_branch1
        ne_old = rearranged_branch[i]
        elems_new[ne, 0] = ne
        elems_new[ne, 1] = elems[ne_old, 1]
        elems_new[ne, 2] = elems[ne_old, 2]
    elems_new = elems_new.astype(int)
    elem_connect = element_connectivity_multi(nodes[:,1:4],elems_new)
    elem_up_new = elem_connect['elem_up']
    index = np.where(elem_up_new[:, 0] == 2)[0][0]
    anast_elem = elems_new[index]
    elem_mod = np.delete(elems_new,index, axis =0)
    elem_mod = np.vstack([elem_mod, anast_elem])
    elem_mod[:,0] = np.arange(0, len(elem_mod))

    return elems_new, elem_mod, anast_elem

def define_radius_by_ordersj(node_loc, elems, system, inlet_elem, inlet_radius, radius_ratio):
    """ This function defines radii in a branching tree by 'order' of the vessel

     Inputs are:
     - node_loc: The nodes in the branching tree
     - elems: The elements in the branching tree
     - system: 'strahler','horsfield' or 'generation' to define vessel order
     - inlet_elem: element number that you want to define as having inlet_radius
     - inlet_radius: the radius of your inlet vessel
     - radius ratio: Strahler or Horsfield type ratio, defines the slope of log(order) vs log(radius)

     Returns:
     -radius of each branch

     A way you might want to use me is:


    This will return:

    >> radius: [ 0.1, 0.06535948 , 0.06535948]"""
    num_elems = len(elems)
    radius = np.zeros(num_elems)  # initialise radius array
    # Evaluate orders in the system
    orders = evaluate_orders(node_loc, elems)
    elem_order = orders[system]
    ne = inlet_elem
    n_max_ord = elem_order[ne]
    radius[ne] = inlet_radius

    for ne in range(0, num_elems):
        radius[ne] = 10. ** (np.log10(radius_ratio) * (elem_order[ne] - n_max_ord) + np.log10(inlet_radius))

    return radius