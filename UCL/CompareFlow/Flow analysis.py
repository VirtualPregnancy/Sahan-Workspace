import placentagen as pg
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.basic import density

from included_analysis_functions import *

sample_number = 'P51'
output_dir = sample_number + '/outputs/'
node_image = np.load(output_dir + 'n_i.npy')
node_microCT = np.load(output_dir + 'n_mCT.npy')
elem_images = np.load(output_dir + 'e_i.npy')
art_elem_images = np.load(output_dir + 'e_i_a.npy')
art_elem_mCT = np.load(output_dir + 'e_mCT_a.npy')

elems_microCT = np.load(output_dir + 'e_mCT.npy')
pressure_images = np.load(output_dir + 'p_i.npy')
pressure_microCT = np.load(output_dir + 'p_mCT.npy')
radii_images = np.load(output_dir + 'r_i.npy')
art_radii_images = np.load(output_dir + 'r_i_a.npy')
art_radii_mCT = np.load(output_dir + 'r_mCT_a.npy')

radii_microCT = np.load(output_dir + 'r_mCT.npy')
flow_images = np.load(output_dir + 'q_i.npy')
art_flow_images = np.load(output_dir + 'q_i_a.npy')
art_flow_mCT = np.load(output_dir + 'q_mCT_a.npy')

flow_microCT = np.load(output_dir + 'q_mCT.npy')
resistance_images = np.load(output_dir + 'R_i.npy')
resistance_microCT = np.load(output_dir + 'R_mCT.npy')

inlet_nidx_images = elem_images[0,1]
inlet_nidx_mCT = elems_microCT[0,1]
inlet_eidx_images = 0
inlet_eidx_mCT = 0

# Subtract the inlet node x, y coordinates from all nodes' x, y
adjusted_nodes_img = node_image.copy()  # Avoid modifying the original array
adjusted_nodes_img[:, 1] -= node_image[inlet_nidx_images][1]  # Adjust x
adjusted_nodes_img[:, 2] -= node_image[inlet_nidx_images][2]  # Adjust y
pg.export_ex_coords(adjusted_nodes_img[:,1:4], 'test', output_dir+'adj_img','exnode')
nodes_max_x = max(node_microCT[:,1])
nodes_max_y = max(node_microCT[:,2])
nodes_max_z = max(node_microCT[:,3])
nodes_min_x = min(node_microCT[:,1])
nodes_min_y = min(node_microCT[:,2])
nodes_min_z = min(node_microCT[:,3])

x_length = nodes_max_x - nodes_min_x
y_length = nodes_max_y - nodes_min_y
z_length = nodes_max_z - nodes_min_z

print('X length:', x_length)
print('Y length:', y_length)
print('Z length:', z_length)

inodes_max_x = max(node_image[:,1])
inodes_max_y = max(node_image[:,2])
inodes_max_z = max(node_image[:,3])
inodes_min_x = min(node_image[:,1])
inodes_min_y = min(node_image[:,2])
inodes_min_z = min(node_image[:,3])

ix_length = inodes_max_x - inodes_min_x
iy_length = inodes_max_y - inodes_min_y
iz_length = inodes_max_z - inodes_min_z

print('X i length:', ix_length)
print('Y i length:', iy_length)
print('Z i length:', iz_length)

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
print('2D')
pg.export_ex_coords(node_image[:,1:4], 'test', output_dir+'art_img','exnode')
pg.export_ex_coords(node_microCT[:,1:4], 'test', output_dir+'art_mCT','exnode')
pg.export_exelem_1d(art_elem_images,'test', output_dir+'art_img')
pg.export_exelem_1d(arterial_elem_microCT,'test', output_dir+'art_mCT')

terminal_img  = pg.calc_terminal_branch(node_image[:,1:4],art_elem_images)
print('mCT')

terminal_mCt  = pg.calc_terminal_branch(node_microCT[:,1:4],arterial_elem_microCT)

EC_img = pg.element_connectivity_1D(arterial_nodes_images[:,1:4],arterial_elem_images)
EC_mCT = pg.element_connectivity_1D(arterial_nodes_microCT[:,1:4],arterial_elem_microCT)
order_img = pg.evaluate_orders(arterial_nodes_images[:,1:4],arterial_elem_images)
order_mCT = pg.evaluate_orders(arterial_nodes_microCT[:,1:4],arterial_elem_microCT)
elems_split_img = strahler_from_input(elem_images,order_img,'strahler',2)
elems_split_mCT = strahler_from_input(elems_microCT,order_mCT,'strahler',2)

split_flows_img = flow_images[elems_split_img[:,0]]
split_flows_mCT = flow_microCT[elems_split_mCT[:,0]]

circle_coords = generate_circle_from_inlet(node_image,elem_images[0,1],30, 100)
circle_nodes = calculate_closest_node(arterial_nodes_images,circle_coords)
circle_elems,circle_flow_images = connected_elems(circle_nodes,elem_images,flow_images)
pg.export_ex_coords(circle_coords, 'circle',output_dir+'CIMG','exnode')
dist_nd_img, dist_elem_img = get_distance_from_inlet(node_image[elem_images[0,1]],node_image,elem_images)
dist_nd_mCT, dist_elem_mCT = get_distance_from_inlet(node_microCT[elems_microCT[0,1]],node_microCT,elems_microCT)
#terminal_elems_img, terminal_flows_img =  connected_elems(terminal_img['terminal_nodes'],terminal_img['terminal_elems'],flow_images)
# Use np.isin to create a boolean mask for rows matching specific node numbers
mask = np.isin(node_image[:, 0], terminal_img['terminal_nodes'])
terminal_nodesdist_img =  dist_nd_img[mask]
mask = np.isin(elem_images[:, 0], terminal_img['terminal_elems'])

terminal_elemsdist_img =  dist_elem_img[mask]
flow_term_img = flow_images[mask]
log_flow_term_img = np.log10(flow_term_img)

mask = np.isin(node_microCT[:, 0], terminal_mCt['terminal_nodes'])
terminal_nodesdist_mCT =  dist_nd_mCT[mask]
mask = np.isin(elems_microCT[:, 0], terminal_mCt['terminal_elems'])
terminal_elemsdist_mCT =  dist_elem_mCT[mask]
flow_term_mCT = flow_microCT[mask]
log_flow_term_mCT = np.log10(flow_term_mCT)

log_flow_img = np.log10(flow_images)
log_flow_mCT = np.log10(flow_microCT)
plt.figure(2)
sns.set_theme(style="ticks")
hexplot = sns.jointplot(x=terminal_elemsdist_mCT, y=log_flow_term_mCT, kind="hex")
plt.axis([20,100,-1.6,-0.5])

plt.xlabel('Distance from inlet')
plt.ylabel('Flow (log)')
plt.title("terminal 2D")
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.ylabel('Number of points')

plt.figure(3)
sns.color_palette("hls",8)

sns.set_theme(style="ticks")
hexplot = sns.jointplot(x=terminal_elemsdist_img, y=log_flow_term_img, kind="hex")
plt.xlabel('Distance from inlet')
plt.axis([20,100,-1.6,-0.5])

plt.ylabel('Flow (log)')
plt.title("terminal 2D")
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.ylabel('Number of points')



plt.figure(4)
sns.color_palette("hls",8)

sns.set_theme(style="ticks")
hexplot = sns.jointplot(x=dist_elem_img, y=log_flow_img, kind="hex")
plt.xlabel('Distance from inlet')
plt.ylabel('Flow (log)')
plt.title("2D")
plt.axis([0,120,-1.6,3.2])

plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.ylabel('Number of points')




plt.figure(5)
sns.set_theme(style="ticks")

hexplot2 = sns.jointplot(x=dist_elem_mCT, y=log_flow_mCT, kind="hex", palette='bright')
plt.title("mCT")
plt.xlabel('Distance from inlet')
plt.ylabel('Flow (log)')
plt.axis([0,120,-1.6,3.2])

plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = hexplot2.fig.add_axes([.85, .25, .03, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.ylabel('Number of points')
plt.show()

circle_coords_mCT = generate_circle_from_inlet(node_microCT,elems_microCT[0,1],30, 100)
circle_nodes_mCT = calculate_closest_node(arterial_nodes_microCT,circle_coords_mCT)
circle_elems_mCT,circle_flow_mCT = connected_elems(circle_nodes_mCT,elems_microCT,flow_microCT)
pg.export_ex_coords(circle_coords_mCT, 'circle',output_dir+'CMCT','exnode')





print('done')