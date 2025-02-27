import placentagen as pg
from skan import draw, Skeleton, summarize
from reprosim.diagnostics import set_diagnostics_level
from reprosim.indices import perfusion_indices, get_ne_radius
from reprosim.geometry import append_units, define_node_geometry, define_1d_element_placenta, define_rad_from_geom, \
    add_matching_mesh, \
    define_capillary_model, define_rad_from_file
from reprosim.repro_exports import export_1d_elem_geometry, export_node_geometry, export_1d_elem_field, \
    export_node_field, export_terminal_perfusion, export_1d_elem_field_grouped, export_1d_elem_geometry_grpd
from reprosim.pressure_resistance_flow import evaluate_prq, calculate_stats
import os

sample_number = 'P51'
input_dir =  sample_number + '/microCT/model/'
output_dir = sample_number + '/outputs/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
Tree_file = input_dir + sample_number + '_full_tree'

####################################################################################
#----------------------------------------------------------------------------------#
#----------------------- Flow and Pressure Generation------------------------------#
#----------------------------------------------------------------------------------#
####################################################################################

print('Beginning flow and pressure simulations (ó﹏ò｡)')

###############################################################
# --------------- Flow simulation setup --------------------- #
###############################################################

set_diagnostics_level(0)  # level 0 - no diagnostics; level 1 - only prints subroutine names (default); level 2 - prints subroutine names and contents of variables
perfusion_indices()
#Load node points in tree
print("Reading elem file", Tree_file)
define_node_geometry(Tree_file + '.ipnode')
#Load elements in tree
define_1d_element_placenta(Tree_file + '.ipelem')

# mesh_type: can be 'simple_tree' or 'full_plus_tube'. Simple_tree is the input
## arterial tree without any special features at the terminal level
# 'full_plus_tube' creates a matching venous mesh and has arteries and
## veins connected by capillary units (capillaries are just tubes represented by an element)
mesh_type = 'full_plus_tube'
#mesh that converges onto the arterial tree
umbilical_elem_option = 'same_as_arterial'
#Boundary condition type: Needs to be either Inlet Pressure and Outlet Pressure or Outlet pressure and inlet flow rate
bc_type = 'flow'  # 'pressure' or 'flow'
#Rheology is constant viscosity. Can also account for the effects of RBC on viscosity
rheology_type = 'constant_visc'
#Vessel type can be rigid or a elastic as a function of diameter
vessel_type = 'rigid'

# define terminal units (this subroutine always needs to be called regardless of mesh_type
append_units()

###############################################################
# --------------- Venous mesh Creation ---------------------- #
###############################################################

#venous mesh creation
umbilical_elements = []
add_matching_mesh(umbilical_elem_option, umbilical_elements)

# define radius by Strahler order in diverging (arterial mesh)
s_ratio = 1.38  # rate of decrease in radius at each order of the arterial tree  1.38
inlet_rad = 1.8  # inlet radius
order_system = 'strahler'
order_options = 'arterial'
name = 'inlet'
Radius_file = Tree_file + '_radius.ipfiel'
define_rad_from_file(Radius_file,order_system,s_ratio)
#define_rad_from_geom(order_system, s_ratio, name, inlet_rad, order_options, '')
# defines radius by Strahler order in converging (venous mesh)
s_ratio_ven = 1.46  # rate of decrease in radius at each order of the venous tree 1.46
inlet_rad_ven = 2.2  # inlet radius
order_system = 'strahler'
order_options = 'venous'
first_ven_no = ''  # number of elements read in plus one
last_ven_no = ''  # 2x the original number of elements + number of connections
define_rad_from_geom(order_system, s_ratio_ven, first_ven_no, inlet_rad_ven, order_options, last_ven_no)

print('Venous mesh created using parameter and order system:', umbilical_elem_option, order_system)
print('Viscosity:', rheology_type)
print('Vessel type:', vessel_type)

###############################################################
# ---------------- Capillary Creation ----------------------- #
###############################################################

num_convolutes = 10  # number of terminal convolute connections
num_generations = 3  # number of generations of symmetric intermediate villous trees
num_parallel = 6  # number of capillaries per convolute
define_capillary_model(num_convolutes, num_generations, num_parallel, 'byrne_simplified')

#Defining boundary conditions. Value at zero is a dummy variable
if bc_type == 'pressure':
    inlet_pressure = 6650  # Pa (~50mmHg)
    outlet_pressure = 2660  # Pa (~20mmHg)
    inlet_flow = 0  # set to 0 for bc_type = pressure;

if bc_type == 'flow':
    inlet_pressure = 0
    outlet_pressure = 2660
    inlet_flow = 4166.7  # mm3/s

####################################################################################
# ---------------- Solve Pressure, resistance and flow rate----------------------- #
####################################################################################

evaluate_prq(mesh_type, bc_type, rheology_type, vessel_type, inlet_flow, inlet_pressure, outlet_pressure)

print('Pressure and flow simulation complete: ৻(  •̀ ᗜ •́  ৻)')

##export geometry
group_name = 'perf_model'
#Full_flow_tree files include venous mesh that matches arterial mesh
export_1d_elem_geometry(output_dir + 'full_flow_tree_microCT_' + sample_number + '.exelem', group_name)
export_node_geometry(output_dir + 'full_flow_tree_microCT_' + sample_number + '.exnode', group_name)

# # export element field for radius
field_name = 'radius_perf'
ne_radius = get_ne_radius()
export_1d_elem_field(ne_radius, output_dir + 'radius_microCT_' + sample_number + '.exelem', group_name, field_name)
# export flow in each element
field_name = 'flow'
export_1d_elem_field(7, output_dir + 'flow_microCT_' + sample_number + '.exelem', group_name, field_name)
# export resistance in each element
field_name = 'resistance'
export_1d_elem_field(8, output_dir + 'resistance_microCT_' + sample_number + '.exelem', group_name, field_name)
#export node field for pressure
field_name = 'pressure_perf'
export_node_field(1, output_dir + 'pressure_microCT_' + sample_number + '.exnode', group_name, field_name)
# Export terminal solution
export_terminal_perfusion(output_dir + 'terminal_microCT' + sample_number + '.exnode', 'terminal_soln')
print('Pressure and flow files exported ৻(  •̀ ᗜ •́  ৻)')
export_1d_elem_geometry_grpd(output_dir + 'art_tree_mCT_' + sample_number + '.exelem', 'Arteries', 'art')
export_1d_elem_field_grouped(ne_radius, output_dir + 'radius_art_mCT_' + sample_number + '.exelem', 'Arteries',
                             'radius', 'art')
export_1d_elem_field_grouped(7, output_dir + 'flow_art_mCT_' + sample_number + '.exelem', 'Arteries', 'flow',
                             'art')