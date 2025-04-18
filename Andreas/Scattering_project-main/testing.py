import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import Matrix_construct
import C2_surface as C2
import plane_wave as PW

def main():
    mu = 1
    omega = 1
    epsilon_int = 2
    epsilon_ext = 1
    k = omega * np.sqrt(epsilon_ext * mu)

    S = C2.sphere(1, np.array([0, 0, 0]), 5)
    A = Matrix_construct.construct_matrix(
        S,
        S.construct_conformal_surface(0.8),
        S.construct_conformal_surface(1.2),
        mu, epsilon_int, epsilon_ext, omega
    )

    PW1 = PW.Plane_wave(np.array([1, 0, 0]), 0, k)
    rhs = Matrix_construct.construct_RHS(S, PW1)

    C = np.linalg.solve(A, rhs)

    plt.plot(np.abs(C))
    plt.show()
    plt.imshow(np.abs(A))

    np.savetxt("A.csv", A, delimiter=",")
    np.savetxt("rhs.csv", rhs, delimiter=",")
    np.savetxt("C.csv", C, delimiter=",")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
