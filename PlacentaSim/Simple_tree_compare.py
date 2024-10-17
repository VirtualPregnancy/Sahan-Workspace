import placentagen as pg
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from traitsui.examples.demo.Advanced.Tabular_editor_with_context_menu_demo import columns

print('Reading node files')

elems_file = pg.import_exelem_tree('full_tree.exelem')
print('Reading resistance')
res = pg.import_exelem_field('resistance_perf.exelem')
elems = elems_file['elems']

nodes = np.unique(elems[:, 1:3])  # Extract unique nodes from n1 and n2
num_nodes = len(nodes)  # Total number of unique nodes (columns of the incidence matrix)

num_elems = len(elems)
n_art_elem = 63607
inlet_pressure  =  6650
outlet_pressure  =  2660
inlet_elem = 0
outlet_elem  = n_art_elem
inlet_node = elems[inlet_elem,1]
outlet_node = elems[outlet_elem,2]

rows = elems[:,1]
col = elems[:,0]
data = np.ones_like(rows)
rows2 = elems[:,2]
data2 = -1*np.ones_like(rows2)
rows_all = np.concatenate((rows,rows2))
col_all = np.concatenate((col,col))
data_all = np.concatenate((data,data2))
# Populate the data for the incidence matrix

# Create the incidence matrix in CSR format
B_csr = sp.csr_matrix((data_all, (rows_all, col_all)), shape=(num_nodes, num_elems))
# Step 1: Create a boolean mask to exclude the rows for inlet and outlet nodes
rows_to_keep = np.ones(B_csr.shape[0], dtype=bool)
rows_to_keep[[inlet_node, outlet_node]] = False

# Step 2: Create the reduced incidence matrix Br_csr by selecting rows to keep
Br_csr = B_csr[rows_to_keep, :]

# Step 3: Create Bp_csr to include only the rows for the inlet and outlet nodes
rows_to_include = [inlet_node, outlet_node]
Bp_csr = B_csr[rows_to_include, :]
Pb = np.array([inlet_pressure, outlet_pressure])

# Output to confirm
print("Reduced incidence matrix (Br_csr) shape:", Br_csr.shape)
print("Boundary incidence matrix (Bp_csr) shape:", Bp_csr.shape)
# Output the incidence matrix
print("Incidence matrix with rows as nodes and columns as bonds:\n", B_csr)
G = sp.diags(1.0/res)
rhs = Br_csr @ (G @ (Bp_csr.transpose() @ Pb))
rhs = -1*rhs
lhs =  Br_csr @ (G @ (Br_csr.transpose()))
P_r = spla.spsolve(lhs, rhs)
P_r = np.insert(P_r,inlet_node,inlet_pressure)
P = np.insert(P_r,outlet_node,outlet_pressure)
pressure_diff = P[elems[:,1].astype(int)] - P[elems[:,2].astype(int)]
Q = pressure_diff/res
print(P_r)