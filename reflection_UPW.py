import numpy as np
import cmath

omega = 1
epsilon0 = 1
n0 = 1
mu0 = 1
k0 = omega * np.sqrt(epsilon0 * mu0)

epsilon_air = epsilon0
epsilon_substrate = 2
n_air = n0
n_substrate = 2
k_air = n_air * k0
k_substrate = n_substrate * k0

E0 = 1

xhat = np.array([1,0,0])
yhat = np.array([0,1,0])
zhat = np.array([0,0,1])


def azimutal_angle(k_inc):
    # Assume that k_inc unit vector
    cos_phi = k_inc[0]
    sin_phi = np.sqrt(1 - cos_phi**2)    # Assume that roation is at most pi

    return cos_phi, sin_phi


def rotate(k, E, cos_phi, sin_phi, inverse = False):
    if inverse:
        Rz = np.array([[cos_phi, sin_phi, 0],
                        [-sin_phi, cos_phi, 0],
                        [0, 0, 1]])
    else:
        Rz = np.array([[cos_phi, -sin_phi, 0],
                        [sin_phi, cos_phi, 0],
                        [0, 0, 1]])

    
    k_rot = Rz @ k
    E_rot = Rz @ E

    return k_rot, E_rot 


def polar_angle(k_inc):
    # Assume that k_inc unit vector
    cos_theta = k_inc[2]
    sin_theta = np.sqrt(1 - cos_theta**2)  # Assuming that wave travels in positive z-direction (from medium 1 to medium 2)

    return cos_theta, sin_theta


# Fresnel coefficients for dielectrics (i.e. mu1 = mu2 = mu0)
def fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc, epsilon1 = epsilon_air, epsilon2 = epsilon_substrate):
    large_sqrt = np.sqrt(epsilon2/epsilon1) * np.sqrt(1 - (epsilon1/epsilon2)*sin_theta_inc**2)
    
    Gamma_r = (cos_theta_inc - large_sqrt)/(cos_theta_inc + large_sqrt)
    Gamma_t = (2*cos_theta_inc)/(cos_theta_inc + large_sqrt)

    return Gamma_r, Gamma_t


def fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc, epsilon1 = epsilon_air, epsilon2 = epsilon_substrate):
    large_sqrt = np.sqrt(epsilon1/epsilon2) * np.sqrt(1 - (epsilon1/epsilon2)*sin_theta_inc**2)
    
    Gamma_r = (-cos_theta_inc + large_sqrt) / (cos_theta_inc + large_sqrt)
    Gamma_t = (2 * np.sqrt(epsilon1/epsilon2) * cos_theta_inc) / (cos_theta_inc + large_sqrt)

    return Gamma_r, Gamma_t


def decomposition(E_inc): #, cos_theta_inc, sin_theta_inc
    cos_beta = E_inc[1] / np.linalg.norm(E_inc)
    sin_beta = np.sqrt(1 - cos_beta**2)

    # E_inc_TE_amp = E_inc[1] / cos_beta
    # E_inc_TM_amp = E_inc[0] / (sin_beta * cos_theta_inc)

    E0 = np.sqrt(np.abs(E_inc[0])**2 + np.abs(E_inc[1])**2 + np.abs(E_inc[2])**2)

    # E_inc_TE = yhat * E0
    # E_inc_TM = (xhat * cos_theta_inc + zhat * sin_theta_inc) * E0

    return E0, cos_beta, sin_beta # E_inc_TE, E_inc_TM



def reflected_field_TE(Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1 = k_air):
    E_ref = lambda r : (yhat * Gamma_r * E0 
                        * np.exp(- complex(0,1) * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)))
    return E_ref


def reflected_field_TM(Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1 = k_air):
    E_ref = lambda r : ((xhat * cos_theta_inc + zhat * sin_theta_inc) * Gamma_r * E0 
                        * np.exp(- complex(0,1) * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)))
    return E_ref


def transmitted_field_TE(Gamma_t, cos_theta_inc, sin_theta_inc, E0, k1 = k_air, k2 = k_substrate):
    sin_theta_trans = k1/k2 * sin_theta_inc
    cos_theta_trans = 1/k2 * np.sqrt(k2**2 - k1**2 * sin_theta_trans**2)

    E_trans = lambda r : (yhat * Gamma_t * E0 
                        * np.exp(- complex(0,1) * k2 * (r[0] * sin_theta_trans + r[2] * cos_theta_trans)))
    return E_trans


def transmitted_field_TM(Gamma_t, cos_theta_inc, sin_theta_inc, E0, k1 = k_air, k2 = k_substrate):
    sin_theta_trans = k1/k2 * sin_theta_inc
    cos_theta_trans = 1/k2 * np.sqrt(k2**2 - k1**2 * sin_theta_trans**2)

    E_trans = lambda r : ((xhat * cos_theta_trans - zhat * sin_theta_trans) * Gamma_t * E0 
                        * np.exp(- complex(0,1) * k2 * (r[0] * sin_theta_trans + r[2] * cos_theta_trans)))
    return E_trans


def driver(k_inc, E_inc, r):
    cos_phi, sin_phi = azimutal_angle(k_inc)

    k_inc_rot, E_inc_rot = rotate(k_inc, E_inc, cos_phi, sin_phi)

    cos_theta_inc, sin_theta_inc = polar_angle(k_inc)

    Gamma_r_TE, Gamma_t_TE = fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc)
    Gamma_r_TM, Gamma_t_TM = fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc)

    E0, cos_beta, sin_beta = decomposition(E_inc)

    
    
