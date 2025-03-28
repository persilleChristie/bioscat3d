import numpy as np

def moveDipoleField(xPrime, rPrime , E, H ):
    # rotate the field with angle between zaxis and r
    cos_theta = rPrime[2] / np.linalg.norm(rPrime)
    sin_theta = np.sqrt(1 - cos_theta**2)

    cos_phi = np.sign(rPrime[1]) * rPrime[0] / np.sqrt(rPrime[0]**2 + rPrime[1]**2)
    sin_phi = np.sqrt(1 - cos_phi**2)

    #rotation matrix for theta around y axis
    Ry = np.array([[cos_theta, 0, sin_theta],
                   [0, 1, 0],
                   [-sin_theta, 0, cos_theta]])

    #rotation matrix for phi around z axis
    Rz = np.array([[cos_phi, -sin_phi, 0],
                   [sin_phi, cos_phi, 0],
                   [0, 0, 1]])
    
    #rotate the field
    E_rot = lambda x: np.dot(Rz, np.dot(Ry, E(x)))
    H_rot = lambda x: np.dot(Rz, np.dot(Ry, H(x)))

    E_prime = lambda x: E_rot(x) - xPrime
    H_prime = lambda x: H_rot(x) - xPrime

    return E_prime, H_prime






