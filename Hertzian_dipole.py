import numpy as np

j = complex(0,1)   # imaginary unit

eta0 = 1
k0 = 1


def electric_field(l, Ie, eta = eta0, k = k0):
    r = lambda x : np.linalg.norm(x)
    
    cos_theta = lambda x : x[2] / r(x)
    sin_theta = lambda x : np.sqrt(1 - cos_theta(x)**2)

    cos_phi = lambda x : np.sign(x[1]) * x[0] / np.sqrt(x[0]**2 + x[1]**2)
    sin_phi = lambda x : np.sqrt(1 - cos_phi(x)**2)

    E_r = lambda x : eta * Ie * l * cos_theta(x) / (2 * np.pi * r(x)**2) * (1 + 1/(j*k*r(x))) * np.exp(-j*k*r(x))
    E_theta = lambda x : (j * eta * Ie * l * sin_theta(x) / (4 * np.pi * r(x)) 
                          * (1 + 1/(j*k*r(x)) - 1/(k*r(x))**2) * np.exp(-j*k*r(x)))
    
    E_x = lambda x : E_r(x) * sin_theta(x) * cos_phi(x) + E_theta(x) * cos_theta(x) * cos_phi(x)
    E_y = lambda x : E_r(x) * sin_theta(x) * sin_phi(x) + E_theta(x) * cos_theta(x) * sin_phi(x)
    E_z = lambda x : E_r(x) * cos_theta(x) - E_theta(x) * sin_theta(x) 

    E = lambda x : np.array([E_x(x), E_y(x), E_z(x)])

    return E


l = 1e-6
Ie = 5

E = electric_field(l, Ie)
x = np.array([1,1,1]) * 0.01

print(E(x))