import numpy as np

j = complex(0,1) # imaginary unit

# TODO: change constants
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

# E0 = 1

xhat = np.array([1,0,0])
yhat = np.array([0,1,0])
zhat = np.array([0,0,1])


def azimutal_angle(k_inc):
    cos_phi = k_inc[0]/np.sqrt(k_inc[0]**2 + k_inc[1]**2)
    sin_phi = np.sqrt(1 - cos_phi**2)    # Assume that roation is at most pi

    return cos_phi, sin_phi


def rotate(k_inc, E_inc, cos_phi, sin_phi):
    Rz = np.array([[cos_phi, -sin_phi, 0],
                    [sin_phi, cos_phi, 0],
                    [0, 0, 1]])

    
    k_rot = Rz @ k_inc
    E_rot = lambda r : Rz @ E_inc(r)

    return k_rot, E_rot 

def rotate_inverse(E_rot, cos_phi, sin_phi):
    Rz_inv = np.array([[cos_phi, sin_phi, 0],
                        [-sin_phi, cos_phi, 0],
                        [0, 0, 1]])
    
    E = lambda r : Rz_inv @ E_rot(r)
    
    return E


def polar_angle(k_rot):
    cos_theta = k_rot[2] / np.sqrt(k_rot[0]**2 + k_rot[2]**2)
    sin_theta = np.sqrt(1 - cos_theta**2)  # Assuming that wave travels in positive z-direction (from medium 1 to medium 2)

    return cos_theta, sin_theta


# Fresnel coefficients for dielectrics (i.e. mu1 = mu2 = mu0)
def fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc, epsilon1 = epsilon_air, epsilon2 = epsilon_substrate):
    large_sqrt = np.sqrt(epsilon2/epsilon1) * np.sqrt(1 - (epsilon1/epsilon2)*sin_theta_inc**2)
    
    Gamma_r = (cos_theta_inc - large_sqrt) / (cos_theta_inc + large_sqrt)
    Gamma_t = (2*cos_theta_inc) / (cos_theta_inc + large_sqrt)

    return Gamma_r, Gamma_t


def fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc, epsilon1 = epsilon_air, epsilon2 = epsilon_substrate):
    large_sqrt = np.sqrt(epsilon1/epsilon2) * np.sqrt(1 - (epsilon1/epsilon2)*sin_theta_inc**2)
    
    Gamma_r = (-cos_theta_inc + large_sqrt) / (cos_theta_inc + large_sqrt)
    Gamma_t = (2 * np.sqrt(epsilon1/epsilon2) * cos_theta_inc) / (cos_theta_inc + large_sqrt)

    return Gamma_r, Gamma_t


def decomposition(E_rot): 
    E0 = lambda r : np.linalg.norm(E_rot(r))

    cos_beta = lambda r : E_rot(r)[1] / E0(r)
    sin_beta = lambda r : np.sqrt(1 - cos_beta(r)**2)

    return E0, cos_beta, sin_beta 



def reflected_field_TE(Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1 = k_air):
    E_ref = lambda r : (yhat * Gamma_r * E0(r) 
                        * np.exp(- j * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)))
    return E_ref


def reflected_field_TM(Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1 = k_air):
    E_ref = lambda r : ((xhat * cos_theta_inc + zhat * sin_theta_inc) * Gamma_r * E0(r) 
                        * np.exp(- j * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)))
    return E_ref


def transmitted_field_TE(Gamma_t, sin_theta_inc, E0, k1 = k_air, k2 = k_substrate):
    sin_theta_trans = k1/k2 * sin_theta_inc
    cos_theta_trans = np.sqrt(1 - sin_theta_trans**2) # 1/k2 * np.sqrt(k2**2 - k1**2 * sin_theta_trans**2)

    E_trans = lambda r : (yhat * Gamma_t * E0(r) 
                        * np.exp(- j * k2 * (r[0] * sin_theta_trans + r[2] * cos_theta_trans)))
    return E_trans


def transmitted_field_TM(Gamma_t, sin_theta_inc, E0, k1 = k_air, k2 = k_substrate):
    sin_theta_trans = k1/k2 * sin_theta_inc
    cos_theta_trans = np.sqrt(1 - sin_theta_trans**2) # 1/k2 * np.sqrt(k2**2 - k1**2 * sin_theta_trans**2)

    E_trans = lambda r : ((xhat * cos_theta_trans - zhat * sin_theta_trans) * Gamma_t * E0(r)
                        * np.exp(- j * k2 * (r[0] * sin_theta_trans + r[2] * cos_theta_trans)))
    return E_trans


def driver(k_inc, E_inc):
    cos_phi, sin_phi = azimutal_angle(k_inc)

    k_rot, E_rot = rotate(k_inc, E_inc, cos_phi, sin_phi)

    cos_theta_inc, sin_theta_inc = polar_angle(k_rot)

    Gamma_r_TE, Gamma_t_TE = fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc)
    Gamma_r_TM, Gamma_t_TM = fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc)

    E0, cos_beta, sin_beta = decomposition(E_rot)

    E_ref_TE = reflected_field_TE(Gamma_r_TE, cos_theta_inc, sin_theta_inc, E0)
    E_ref_TM = reflected_field_TM(Gamma_r_TM, cos_theta_inc, sin_theta_inc, E0)
    E_ref_rot = lambda r : cos_beta(r) * E_ref_TE(r) + sin_beta(r) * E_ref_TM(r)

    E_trans_TE = transmitted_field_TE(Gamma_t_TE, sin_theta_inc, E0)
    E_trans_TM = transmitted_field_TM(Gamma_t_TM, sin_theta_inc, E0)
    E_trans_rot = lambda r : cos_beta(r) * E_trans_TE(r) + sin_beta(r) * E_trans_TM(r)

    E_ref = rotate_inverse(E_ref_rot, cos_phi, sin_phi)
    E_trans = rotate_inverse(E_trans_rot, cos_phi, sin_phi)

    E_tot = lambda r : (E_trans(r)) * (r[2] > 0) + (E_ref(r) + E_inc(r)) * (r[2] < 0)

    return E_tot

    


k_inc = np.array([1,0,2])
E_inc = lambda r : np.array([2,-1,-1])
r = np.array([2,2,2])
R = [np.random.rand(3) for _ in range(10)]

E_tot = driver(k_inc, E_inc)

print([E_tot(R[i]) for i in range(10)])