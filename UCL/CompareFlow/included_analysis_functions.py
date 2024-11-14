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


