from fetoflow import *
import pandas as pd
import polyscope as ps

def main():

    nodes = read_nodes('full_tree_P51.ipnode')
    elems = read_elements('full_tree_P51.ipelem')
    outlet_pressure, inlet_flow = 2660, 4166.7
    bcs = generate_boundary_conditions(outlet_pressure=outlet_pressure,inlet_flow=inlet_flow)

    # define other required geometric features (radii and decay factors)
    umbilical_artery_radius, decay_factor = 1.8/1000, 1.38
    umbilical_vein_radius, decay_factor_vein = 4.0/1000 , 1.46
    arteries_only = False # this should rarely be true
    viscosity_type = 'constant' # can also be 'pries_network' or 'pries_vessel' if wanting to incorporate radius-dependence

    # Generate the di-graph & calculate the resistances based on the viscosity
    print("Creating Geometry")
    G = create_geometry(nodes, elems, umbilical_artery_radius, decay_factor,umbilical_vein_radius, decay_factor_vein, arteries_only=arteries_only)
    print("Calculating Resistance")
    G = calculate_resistance(G, viscosity_model=viscosity_type)
    print("Calculating Matrices")
    A, b, bc_export = create_small_matrices(G, bcs, branching_angles=False)
    print("Solving for Pressures and Flows")
    p, q = solve_small_system(A, b, G, bc_export)
    for k in q:
        q[k] *= 1e9
    print('hi')

if __name__ == '__main__':
    main()