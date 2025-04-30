#-----------------------------------------------------------------------------
#                                   Imports
#-----------------------------------------------------------------------------
import Hertzian_dipole_speedup as HD
import C2_surface
import plane_wave_speedup as PW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json


def construct_RHSs(Surface, propagation_vectors, polarizations, epsilon_air, mu, omegas):
    '''
    Constructs RHS matrix for multiple incident plane waves, ready for least squares solvers.

    Output:
        rhs_matrix: (4*N) x M array
                    - Each column corresponds to the RHS for one plane wave.
                    - Each block of N rows corresponds to:
                        [0:N]   → -E ⋅ tau1
                        [N:2N]  → -E ⋅ tau2
                        [2N:3N] → -H ⋅ tau1
                        [3N:4N] → -H ⋅ tau2

    Input:
        Surface: object with attributes
            - points: Nx3 array
            - tau1: Nx3 array (tangent vector 1)
            - tau2: Nx3 array (tangent vector 2)
        propagation_vectors: Mx3 array
        polarizations: M array
        epsilon_air: scalar
        mu: scalar
        omegas: M array
    '''
    points = Surface.points
    tau1 = Surface.tau1
    tau2 = Surface.tau2
    
    # Evaluate fields for all M plane waves
    planewaves = PW.Plane_wave(propagation_vectors, polarizations, epsilon_air, mu, omegas)
    E_all, H_all = planewaves.evaluate_at_points(points)  # (2, M, N, 3)

    # Compute tangential components → shape (M, N)
    b1 = -np.einsum("mnj,nj->mn", E_all, tau1)
    b2 = -np.einsum("mnj,nj->mn", E_all, tau2)
    b3 = -np.einsum("mnj,nj->mn", H_all, tau1)
    b4 = -np.einsum("mnj,nj->mn", H_all, tau2)

    b=np.hstack([b1,b2,b3,b4]) #Stack columns to get a (M_planewaves,4*N points matrix)
    rhs_matrix=b.T # Transpose it to get the desired (4*N, M) matrix

    return rhs_matrix

def construct_sub_column(dipoles, Surface):
    '''
    Computes a column block of the MAS matrix corresponding to a set of dipoles.

    Input:
        dipoles
        Surface: Surface object with:
            - points: (N, 3)
            - tau1, tau2: (N, 3) tangent vectors
            - N: number of test points

    Output:
        (4*N, M) matrix — tangential components of E and H fields from each dipole
    '''
    points = Surface.points
    tau1 = Surface.tau1
    tau2 = Surface.tau2
    N = points.shape[0]

    # Evaluate E and H fields
    E_all, H_all = dipoles.evaluate_at_points(points)  # (M_dipoles, N_points, 3)

    # Compute projections using einsum
    E_tau1 = np.einsum('nmj,mj->nm', E_all, tau1)  # (M_dipoles, N_points)
    E_tau2 = np.einsum('nmj,mj->nm', E_all, tau2)
    H_tau1 = np.einsum('nmj,mj->nm', H_all, tau1)
    H_tau2 = np.einsum('nmj,mj->nm', H_all, tau2)

    # Stack the projections: shape (4*N, M_dipoles)
    sub_column = np.vstack([
        E_tau1.T,
        E_tau2.T,
        H_tau1.T,
        H_tau2.T
    ])

    return sub_column

def construct_matrix(Surface, inneraux, outeraux, mu, air_epsilon, scatter_epsilon, omega):
    '''
    Constructs the full 4x4 block MAS matrix using HertzianDipole class instances.

    Returns:
        MAS matrix: shape (4*M, 4*N)
        Dipole instances: intDP1, intDP2, extDP1, extDP2
    '''
    #----------------------------------------
    # Extract geometry
    #----------------------------------------
    inner_points = inneraux.points
    inner_tau1 = inneraux.tau1
    inner_tau2 = inneraux.tau2
    outer_points = outeraux.points
    outer_tau1 = outeraux.tau1
    outer_tau2 = outeraux.tau2

    #----------------------------------------
    # Create Hertzian Dipole instances
    #----------------------------------------
    intDP1 = HD.Hertzian_Dipole(inner_points, inner_tau1, mu, air_epsilon, omega)
    intDP2 = HD.Hertzian_Dipole(inner_points, inner_tau2, mu, air_epsilon, omega)
    extDP1 = HD.Hertzian_Dipole(outer_points, outer_tau1, mu, scatter_epsilon, omega)
    extDP2 = HD.Hertzian_Dipole(outer_points, outer_tau2, mu, scatter_epsilon, omega)
    #----------------------------------------
    # Construct block columns
    #----------------------------------------
    
    Col1 = construct_sub_column(intDP1, Surface)
    Col2 = construct_sub_column(intDP2, Surface)
    Col3 = construct_sub_column(extDP1, Surface)
    Col4 = construct_sub_column(extDP2, Surface)

    #----------------------------------------
    # Assemble MAS matrix
    #----------------------------------------
    MAS = np.column_stack((Col1, Col2, Col3, Col4))

    return MAS, intDP1, intDP2, extDP1, extDP2
    
def Construct_solve_MAS_system(Scatter_information, Incident_information, plot=False, reduce_grid=True):
    '''
    Solves the MAS system for multiple plane wave excitations (same omega, mu, epsilon).

    Output:
        int_coeffs: [C1, C2] arrays of shape (M, R) for interior dipoles in tau1 and tau2 (M dipoles, R planewaves)
        ext_coeffs: [C3, C4] arrays of shape (N, M) for exterior dipoles in tau1 and tau2 (M dipoles, R planewaves)
        InteriorDipoles: [intDP1, intDP2] — input dipole data
        ExteriorDipoles: [extDP1, extDP2] — input dipole data

    Input:
        Scatter_information: dict containing surface + permittivity/permeability info
        Incident_information: dict containing *lists* of propagation_vectors and polarizations
                              and shared epsilon, mu, omega
    '''
    #-------------------------------------------------------------
    # Unpack scattering info
    #-------------------------------------------------------------
    Surface = Scatter_information['Surface']
    inneraux = Scatter_information['inneraux']
    outeraux = Scatter_information['outeraux']
    scatter_epsilon = Scatter_information['epsilon']
    scatter_mu = Scatter_information['mu']

    #-------------------------------------------------------------
    # Unpack incident info
    #-------------------------------------------------------------
    propagation_vectors = Incident_information['propagation_vectors']  # shape (R, 3)
    polarizations = Incident_information['polarizations']              # shape (R,)
    epsilon_air = Incident_information['epsilon']
    incident_mu = Incident_information['mu']
    omega = Incident_information['omega']
    lam = Incident_information['lambda']

    #-------------------------------------------------------------
    # Reduce surface information to necessary wavelength
    #-------------------------------------------------------------

    if reduce_grid:
        Surface, inneraux, outeraux = C2_surface.Set_dipoles_pr_WL(Surface,inneraux,outeraux,lam)
    M=np.shape(inneraux.points)[0]
    N=np.shape(Surface.points)[0]
    R=len(polarizations)
    print(f"M: {M}, N: {N}, R: {R}")
    #-------------------------------------------------------------
    # Construct RHS matrix and MAS matrix
    #-------------------------------------------------------------
    con_time = time.time()
    RHS_matrix = construct_RHSs(Surface, propagation_vectors, polarizations, epsilon_air, incident_mu, omega)
    construct_matrix(Surface,inneraux,outeraux,scatter_mu,epsilon_air,scatter_epsilon,omega)
    MAS_matrix, intDP1, intDP2, extDP1, extDP2 = construct_matrix(
        Surface, inneraux, outeraux, scatter_mu, epsilon_air, scatter_epsilon, omega
    )
    print(f"Number of RHS: {np.shape(RHS_matrix)[1]}")
    print(f"Construction time: {time.time() - con_time:.3f} s")
    #print(f"Matrix shape: {MAS_matrix.shape}, RHS shape: {RHS_matrix.shape}")

    #-------------------------------------------------------------
    # Solve system for all RHS
    #-------------------------------------------------------------

    sol_start = time.time()
    C_matrix, *_ = np.linalg.lstsq(MAS_matrix, RHS_matrix, rcond=None) #solution size 4MxR
    C1 = C_matrix[:M]
    C2 = C_matrix[M:2*M]
    C3 = C_matrix[2*M:3*M]
    C4 = C_matrix[3*M:]
    print(f"Solution time: {time.time() - sol_start:.3f} s")

    #-------------------------------------------------------------
    # Optional plotting
    #-------------------------------------------------------------
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        im0 = axs[0].imshow(np.abs(MAS_matrix), aspect='auto', cmap='viridis')
        axs[0].set_title('abs(MAS_matrix)')
        plt.colorbar(im0, ax=axs[0])
        
        axs[1].imshow(np.abs(RHS_matrix), cmap='viridis')
        axs[1].set_title('abs(RHS matrix)')
        axs[1].set_xlabel('Incident Wave Index')
        
        axs[2].imshow(np.abs(C_matrix), cmap='viridis')
        axs[2].set_title('abs(Solution C)')
        axs[2].set_xlabel('Incident Wave Index')
        
        plt.tight_layout()
        plt.show()

    #-------------------------------------------------------------
    # Return values
    #-------------------------------------------------------------
    int_coeffs = [C1, C2]  # each shape (N, M)
    ext_coeffs = [C3, C4]
    InteriorDipoles = [intDP1, intDP2]
    ExteriorDipoles = [extDP1, extDP2]

    return int_coeffs, ext_coeffs, InteriorDipoles, ExteriorDipoles

def compute_scattered_field_at_point(points, int_coeff, InteriorDipoles):
    """
    Compute scattered EM fields at given points due to interior dipoles
    for multiple sets of dipole coefficients (i.e., multiple incident conditions).

    Parameters:
        points : (N, 3) array — evaluation points, (N points)
        int_coeff : [C_1, C_2] — each of shape (M, R), (M dipoles, R incident coefficients)
        InteriorDipoles : [intDP1, intDP2] — dipole objects 

    Returns:
        (2, R, N, 3) array — scattered E and H fields
    """
    C_1, C_2 = int_coeff                      # Shape: (M, R)
    intDP1, intDP2 = InteriorDipoles

    # Evaluate dipole fields: shape (2, M, N, 3)
    evals1 = intDP1.evaluate_at_points(points)  # shape (2, M, N, 3)
    evals2 = intDP2.evaluate_at_points(points)  # shape (2, M, N, 3)

    # Use einsum to perform summation over dipoles with coefficients
    # 'a m n d, m r -> a r n d' where:
    # a = 2 (E, H), m = M (dipoles), n = N (points), d = 3 (vector components), r = R (inputs)
    total_1 = np.einsum('amnd,mr->arnd', evals1, C_1)
    total_2 = np.einsum('amnd,mr->arnd', evals2, C_2)

    # Add contributions from both dipole groups
    total_field = total_1 + total_2  # shape: (2, R, N, 3)
    return total_field

def compute_flux_integral_scattered_field(plane, int_coeff, InteriorDipoles,plot_first_integrand=False):
    '''
    Computes the average power (flux) integral for the scattered field for multiple RHSs.

    Input:
        plane: A C2_object (with .points and .normals)
        int_coeff: List of dipole weights [C_1, C_2], each (M_dipoles, R_incident configs)
        dipoles: List of dipole classes [intDP1, intDP2]

    Output:
        flux_values: Array of shape (R,) — power flux per RHS
    '''
    int_start=time.time()
    #---------------------------------------------------------------------
    # Extract geometry
    #---------------------------------------------------------------------
    points = plane.points             # (N, 3)
    normals = plane.normals           # (N, 3)

    dx = np.linalg.norm(points[1] - points[0])
    dA = dx * dx                      # Scalar area element (uniform)

    #---------------------------------------------------------------------
    # Evaluate scattered fields: (2, M, N, 3)
    #---------------------------------------------------------------------
    E, H = compute_scattered_field_at_point(points, int_coeff, InteriorDipoles) #E and H (R,N,3) each
    
    R, N , _ = np.shape(E)
    Cross = 0.5 * np.cross(E, np.conj(H)) # (R,N,3)
    integrands = np.einsum("rnk,nk->rn", Cross,normals) # (R,N)

    if plot_first_integrand:
        first_integrand=integrands[0,:]
        N=int(np.sqrt(N))
        plt.imshow(np.abs(np.reshape(first_integrand,(N,N))))
        plt.show()
    #Integral calculation
    integrals = np.einsum("rn -> r", integrands*dA)    # (M,)
    print(f"integration_time: {time.time()-int_start}")
    return integrals  # Return real-valued power flux

def Single_scatter_solver(Scatter_information, Incident_configurations, options):
    """
    Solves the forward problem for a specified scatterer and multiple sets of incident waves sorted by wavelength.

    Returns
    -------
    flux_integrals : np.ndarray
        Real part of the flux integrals for each plane wave configuration (flattened).
    """
    show_MAS         = options.get('show_MAS', False)
    show_power_curve = options.get('Show_power_curve', False)
    plane_z          = options.get('plane_location', None)

    Surface  = Scatter_information['Surface']
    pts      = Surface.points
    x        = pts[:,0]
    a, b     = np.min(x), np.max(x)
    N_grid   = int(np.sqrt(pts.shape[0]))  # assume N×N grid

    if plane_z is None:
        plane_z = 5 * pts[:,2].max()

    Plane = C2_surface.generate_plane_xy(plane_z, a, b, 10)

    all_flux = []

    for idx, inc_lam in enumerate(Incident_configurations):
        print(f"\nComputing incident information number {idx+1}/{len(Incident_configurations)}, wavelength: {inc_lam['lambda']:.4f}")
        total_time = time.time()

        int_coeffs, _, InteriorDipoles, _ = Construct_solve_MAS_system(
            Scatter_information,
            inc_lam,
            plot=show_MAS
        )

        power_ints = compute_flux_integral_scattered_field(
            plane=Plane,
            InteriorDipoles=InteriorDipoles,
            int_coeff=int_coeffs
        )  # shape: (M_block,)

        all_flux.append(np.real(power_ints))  # collect real parts

        if show_power_curve:
            plt.figure()
            plt.plot(np.abs(power_ints), marker='o')
            plt.xlabel('Incident index')
            plt.ylabel('Power integral')
            plt.title('Scattered Power vs. Incident Wave')
            plt.tight_layout()
            plt.show()

        print(f"total time: {time.time() - total_time:.2f} seconds")

    return np.concatenate(all_flux)  # shape: (total_M_block,)

def Multiple_scatter_solver(Scatter_configurations, Incident_configurations, options):
    """
    Solves the forward problem for a set of scatterers and incident wave configurations.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing averaged flux integrals across all scatterers.
    """
    all_fluxes = []
    completion_time=time.time()
    for i,Scatter in enumerate(Scatter_configurations):
        print(f"\n\nSolving realization {i+1}/{len(Scatter_configurations)}")
        flux = Single_scatter_solver(Scatter, Incident_configurations, options)
        all_fluxes.append(flux)
    # Stack into shape (n_scatterers, total_M_block)
    flux_array = np.stack(all_fluxes, axis=0)  # shape: (S, T)
    mean_flux = np.mean(flux_array, axis=0)    # shape: (T,)
    # Build metadata from the first scatterer config
    first_config = Incident_configurations
    records = []
    for inc in first_config:
        props = inc['propagation_vectors']
        pols  = inc['polarizations']
        omega = inc['omega']
        lam   = inc['lambda']
        freq  = omega / (2*np.pi)

        for j in range(props.shape[0]):
            records.append({
                'propagation_vector': props[j,:],
                'polarization'      : pols[j],
                'wavelength'        : lam,
                'frequency'         : freq,
                'mean_simulated_fluxintegral': mean_flux[len(records)]
            })
    config=0
    for inc in Incident_configurations:
        config+=len(inc['polarizations'])
    completion_time=time.time()-time.time()
    print(f"\n\nCompletion time: {completion_time}")
    print(f"Number of forward problems solved: {len(Scatter_configurations)*config}")
    print(f"Average time pr forward solve: {completion_time/(len(Scatter_configurations)*config)}")
    return pd.DataFrame.from_records(records)


def bump_test(width=1,resol=160): 
    #----------------------------------------
    #       Surface creation
    # ---------------------------------------  
    a,b=-width,width
    X0=np.linspace(a,b,resol)
    Y0=np.linspace(a,b,resol)
    X,Y=np.meshgrid(X0,Y0)
    bump = lambda x,y,x0,y0,height,sigma: height*np.exp(
        -( (x-x0)**2 + (y-y0)**2 ) / (2*sigma**2)
    )
    f = lambda x,y: (
                    bump(x,y,-0.20073580984422001,0.7211428902558659,0.31959818254342154,0.49932924209851826)+
                    bump(x,y,-0.5503701752921016,-0.5504087674620758,0.11742508365045984,0.6330880728874675) +
                    bump(x,y,0.16178401878913407,0.3329161244736727,0.10617534828874074,0.6849549260809971) 
                     )
    Z=f(X,Y)
    point_cloud,tau1,tau2,normals,mean_curvature=C2_surface.compute_geometric_data(X,Y,Z,(width-(-width))/resol)
    inner_cloud=C2_surface.generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,-0.86)
    outer_cloud=C2_surface.generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,0.86)
        
    Surface=C2_surface.C2_surface(point_cloud,normals,tau1,tau2)
    inneraux=C2_surface.C2_surface(inner_cloud,normals,tau1,tau2)
    outeraux=C2_surface.C2_surface(outer_cloud,normals,tau1,tau2)

    #---------------------------------------------
    #           Scattering information
    #---------------------------------------------
    scatter_epsilon=2.56
    mu=1
    Scatterinformation={'Surface': Surface,'inneraux': inneraux, 'outeraux': outeraux,'epsilon': scatter_epsilon,'mu': mu}
    
    #---------------------------------------------
    #           Incident information
    #---------------------------------------------
    Incidentinformations=[]
    for i in range(100):
        number=2
        propagation_vector = np.tile([0, 0, -1], (number, 1))
        polarization=np.linspace(0,np.pi/2,number)
        wavelength=0.5
        epsilon_air=1
        #wavelength=1.5
        omega=2*np.pi/wavelength
        Incidentinformations.append(
            {'propagation_vectors': propagation_vector, 'polarizations': polarization, 'epsilon': epsilon_air, 'mu': mu,'lambda': wavelength, 'omega':omega}
        )
    options = {
    'show_MAS'         : False,
    'plane_location'   : None,   # auto = 5×max height
    'Show_power_curve' : False
    }
    #Single_scatter_solver(Scatterinformation,Incidentinformations,options)
    df=Multiple_scatter_solver([Scatterinformation],Incidentinformations,options)
    print(df)

def create_surface_and_scattering_info_from_json(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)

    width = params['halfWidth_x']
    resol = params['resolution']
    alpha = params['alpha']
    bump_params = params['bumpData']
    scatter_epsilon = params['epsilon1']
    mu = 1  # Assumed constant

    # -------------------------
    # Surface creation
    # -------------------------
    a, b = -width, width
    X0 = np.linspace(a, b, resol)
    Y0 = np.linspace(a, b, resol)
    X, Y = np.meshgrid(X0, Y0)

    def bump(x, y, x0, y0, height, sigma):
        return height * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def surface_function(x, y):
        return sum(
            bump(x, y, b['x0'], b['y0'], b['height'], b['sigma'])
            for b in bump_params
        )

    Z = surface_function(X, Y)

    dx = (b - a) / resol
    point_cloud, tau1, tau2, normals, mean_curvature = C2_surface.compute_geometric_data(X, Y, Z, dx)
    inner_cloud = C2_surface.generate_curvature_scaled_offset(point_cloud, normals, mean_curvature, -alpha)
    outer_cloud = C2_surface.generate_curvature_scaled_offset(point_cloud, normals, mean_curvature, alpha)

    Surface = C2_surface.C2_surface(point_cloud, normals, tau1, tau2)
    inneraux = C2_surface.C2_surface(inner_cloud, normals, tau1, tau2)
    outeraux = C2_surface.C2_surface(outer_cloud, normals, tau1, tau2)

    Scatterinformation = {
        'Surface': Surface,
        'inneraux': inneraux,
        'outeraux': outeraux,
        'epsilon': scatter_epsilon,
        'mu': mu
    }

    # -------------------------
    # Incident information
    # -------------------------
    k = np.array(params['k'])
    betas = np.array(params['betas'])
    wavelength = 2 * np.pi / params['omega']
    epsilon_air = 1
    number = len(betas)

    propagation_vector = np.tile(k, (number, 1))
    polarization = betas
    omega = params['omega']

    Incidentinformations = [{
        'propagation_vectors': propagation_vector,
        'polarizations': polarization,
        'epsilon': epsilon_air,
        'mu': mu,
        'lambda': wavelength,
        'omega': omega
    }]
    options = {
        'show_MAS': True,
        'plane_location': None,
        'Show_power_curve': False
    }

    Single_scatter_solver(Scatterinformation,Incidentinformations,options)

create_surface_and_scattering_info_from_json('../../MeepTests/SetupTests/SurfaceData/surfaceParams.json')
#bump_test()