import placentagen as pg
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.basic import density

from included_analysis_functions import *

sample_number = 'P51'
output_dir = sample_number + '/outputs/'
node_image = np.load(output_dir + 'n_i.npy')
node_microCT = np.load(output_dir + 'n_mCT.npy')
elem_images = np.load(output_dir + 'e_i.npy')
elems_microCT = np.load(output_dir + 'e_mCT.npy')
pressure_images = np.load(output_dir + 'p_i.npy')
pressure_microCT = np.load(output_dir + 'p_mCT.npy')
radii_images = np.load(output_dir + 'r_i.npy')
radii_microCT = np.load(output_dir + 'r_mCT.npy')
flow_images = np.load(output_dir + 'q_i.npy')
flow_microCT = np.load(output_dir + 'q_mCT.npy')
resistance_images = np.load(output_dir + 'R_i.npy')
resistance_microCT = np.load(output_dir + 'R_mCT.npy')
inlet_nidx_images = 0
inlet_nidx_mCT = 0
inlet_eidx_images = 0
inlet_eidx_mCT = 0

x_im = np.where(pressure_images[:,1] == 2660) #pressure used for outlet boundary
x_mCT = np.where(pressure_microCT[:,1] == 2660) #pressure used for outlet boundary
outlet_nidx_images = x_im[0][0]
outlet_nidx_mCT = x_mCT[0][0]
x_im = np.where(elem_images[:,2] == outlet_nidx_images) #pressure used for outlet boundary
x_mCT = np.where(elems_microCT[:,2] == outlet_nidx_mCT) #pressure used for outlet boundary
outlet_eidx_images = x_im[0][0]
outlet_eidx_mCT = x_mCT[0][0]

total_R_images = (pressure_images[inlet_nidx_images,1] - pressure_images[outlet_nidx_images,1])/flow_images[inlet_eidx_images]
total_R_mCT = (pressure_microCT[inlet_nidx_mCT,1] - pressure_microCT[outlet_nidx_mCT,1])/flow_microCT[inlet_eidx_mCT]

print('Total Resistance 2D photos:', total_R_images)
print('Total Resistance microCT:', total_R_mCT)

arterial_nodes_images = node_image[0:outlet_nidx_images,:]
arterial_nodes_microCT = node_microCT[0:outlet_nidx_mCT,:]
arterial_elem_images = elem_images[0:outlet_eidx_images,:]
arterial_elem_microCT = elems_microCT[0:outlet_eidx_mCT,:]

EC_img = pg.element_connectivity_1D(arterial_nodes_images[:,1:4],arterial_elem_images)
EC_mCT = pg.element_connectivity_1D(arterial_nodes_microCT[:,1:4],arterial_elem_microCT)
order_img = pg.evaluate_orders(arterial_nodes_images[:,1:4],arterial_elem_images)
order_mCT = pg.evaluate_orders(arterial_nodes_microCT[:,1:4],arterial_elem_microCT)
elems_split_img = strahler_from_input(elem_images,order_img,'strahler',1)
elems_split_mCT = strahler_from_input(elems_microCT,order_mCT,'strahler',1)

split_flows_img = flow_images[elems_split_img[:,0]]
split_flows_mCT = flow_microCT[elems_split_mCT[:,0]]
fig, axes = plt.subplots(1,2)
axes[0].hist(split_flows_img, bins=10)
axes[1].hist(split_flows_mCT,bins=10)
plt.tight_layout()
plt.show()


print('done')