for ne in range(num_elems - 1, -1, -1):

    n_horsfield = np.maximum(horsfield[ne], 1)
    n_children = elem_down[ne][0]
    if n_children == 1:
        if generation[elem_down[ne][1]] == 0:
            n_children = 0
    temp_strahler = 0
    strahler_add = 1
    if n_children >= 2:  # Bifurcation downstream
        temp_strahler = strahler[elem_down[ne][1]]  # first daughter
        for noelem in range(1, n_children + 1):
            ne2 = elem_down[ne][noelem]
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
        ne2 = elem_down[ne][1]  # element no of daughter
        n_horsfield = horsfield[ne2]
        strahler_add = strahler[ne2]
    horsfield[ne] = n_horsfield
    strahler[ne] = temp_strahler + strahler_add
