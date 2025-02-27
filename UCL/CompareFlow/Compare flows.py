import placentagen as pg
import pandas as pd
import numpy as np
from tempfile import TemporaryFile
sample_number = 'P51'
output_dir = sample_number + '/outputs/'

nodes_images_file =  pg.import_exnode_tree(output_dir + 'full_flow_tree_images_' + sample_number + '.exnode')
nodes_image = nodes_images_file['nodes']
nodes_microCT_file =  pg.import_exnode_tree(output_dir + 'full_flow_tree_microCT_' + sample_number + '.exnode')
nodes_microCT = nodes_microCT_file['nodes']

elems_images_file =  pg.import_exelem_tree(output_dir + 'full_flow_tree_images_' + sample_number + '.exelem')
elems_images = elems_images_file['elems']
elems_microCT_file =  pg.import_exelem_tree(output_dir + 'full_flow_tree_microCT_' + sample_number + '.exelem')
elems_microCT = elems_microCT_file['elems']
art_elems_images_file = pg.import_exelem_tree(output_dir + 'art_tree_' + sample_number + '.exelem')
art_elems_images = art_elems_images_file['elems']
art_elems_mCT_file = pg.import_exelem_tree(output_dir + 'art_tree_mCT_' + sample_number + '.exelem')
art_elems_mCT = art_elems_images_file['elems']

pressure_images_file =  pg.import_exnode_tree(output_dir + 'pressure_images_' + sample_number + '.exnode')
pressure_images = pressure_images_file['nodes']
pressure_microCT_file =  pg.import_exnode_tree(output_dir + 'pressure_microCT_' + sample_number + '.exnode')
pressure_microCT = pressure_microCT_file['nodes']

radii_images = pg.import_exelem_field(output_dir + 'radius_images_' + sample_number + '.exelem')
radii_microCT = pg.import_exelem_field(output_dir + 'radius_microCT_' + sample_number + '.exelem')
radii_art_images = pg.import_exelem_field(output_dir + 'radius_art_perf_' + sample_number + '.exelem')
radii_art_mCT = pg.import_exelem_field(output_dir + 'radius_art_mCT_' + sample_number + '.exelem')

flow_images = pg.import_exelem_field(output_dir + 'flow_images_' + sample_number + '.exelem')
flow_microCT = pg.import_exelem_field(output_dir + 'flow_microCT_' + sample_number + '.exelem')
flow_art_images = pg.import_exelem_field(output_dir + 'flow_art_perf_' + sample_number + '.exelem')
flow_art_mCT = pg.import_exelem_field(output_dir + 'flow_art_mCT_' + sample_number + '.exelem')

resistance_images = pg.import_exelem_field(output_dir + 'resistance_images_' + sample_number + '.exelem')
resistance_microCT = pg.import_exelem_field(output_dir + 'resistance_microCT_' + sample_number + '.exelem')
outfile = TemporaryFile()
np.save(output_dir + 'n_i', nodes_image)
np.save(output_dir + 'n_mCT', nodes_microCT)
np.save(output_dir + 'e_i', elems_images)
np.save(output_dir + 'e_mCT', elems_microCT)
np.save(output_dir + 'p_i', pressure_images)
np.save(output_dir + 'p_mCT', pressure_microCT)
np.save(output_dir + 'r_i', radii_images)
np.save(output_dir + 'r_mCT', radii_microCT)
np.save(output_dir + 'q_i', flow_images)
np.save(output_dir + 'q_mCT', flow_microCT)
np.save(output_dir + 'R_i',resistance_images)
np.save(output_dir + 'R_mCT', resistance_microCT)
np.save(output_dir + 'r_i_a', radii_art_images)
np.save(output_dir + 'e_i_a', art_elems_images)
np.save(output_dir + 'q_i_a', flow_art_images)

np.save(output_dir + 'r_mCT_a', radii_art_mCT)
np.save(output_dir + 'e_mCT_a', art_elems_mCT)
np.save(output_dir + 'q_mCT_a', flow_art_mCT)

print('done')




#node_images_df.to_csv('nodedata_images.csv',index=False)
