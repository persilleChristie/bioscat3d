import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set tolerance for 0 values
tol = 1e-16


############# PN ############
# Load system matrix
df = pd.read_csv('../Power_int_test/FilesCSV/matrix_A_simple.csv', sep=",", header=None)

df = df.astype('string')

# Calculate condition number and number of zeros
A = df.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()
# _, S, _ = np.linalg.svd(A)
# print(f"Condition number PN: {S[0]/S[-1]}")

# print(f"Total zeros: {np.sum(A<tol)}")
# print(f"Actual zeros: {np.sum(A == 0)}")

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
im0 = axs.imshow(A, cmap='viridis')
axs.set_title("PN actual 0")
plt.colorbar(im0, ax=axs)

# Load RHS
df1 = pd.read_csv('../ForwardSolver/FilesCSV/vector_b_simple.csv', sep=",", header=None)

df1 = df1.astype('string')

b = df1.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()


# Load solution
df1 = pd.read_csv('../ForwardSolver/FilesCSV/solution_y_simple.csv', sep=",", header=None)

df1 = df1.astype('string')

y = df1.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

############# ANDREAS #############
# Load system matrix
df1 = pd.read_csv('A_Andreas.csv', sep=",", header=None, dtype=str)
# Convert to complex numbers
AA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

# Calculate condition number and number of zeros
_, S, _ = np.linalg.svd(AA)
print(f"Condition number A:  {S[0]/S[-1]}")
print(f"Total zeros: {np.sum(AA<tol)}")

# Load solution
df1 = pd.read_csv('y_Andreas.csv', sep=",", header=None, dtype=str)
yA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

# Load RHS
df1 = pd.read_csv('b_Andreas.csv', sep=",", header=None, dtype=str)
bA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

# ############# COMPARISONS ##############
# # How often do we agree on zeros?
# print(f"In same places?: {np.sum((AA<tol) * (A < tol))}")

# # Where do we disagree on zeros?
# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# im0 = axs.imshow((A < tol) * (AA > tol), cmap='viridis')
# axs.set_title("PN 0, Andreas not 0")
# plt.colorbar(im0, ax=axs)

# # Where do we agree?
# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# im0 = axs.imshow((A < tol) * (AA < tol), cmap='viridis')
# axs.set_title("PN 0, Andreas 0")
# plt.colorbar(im0, ax=axs)


# Comparisons of absolute values
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

im0= axs[0].imshow(np.abs(A), cmap='viridis')
axs[0].set_title('abs(A) PN')
plt.colorbar(im0, ax=axs[1])

im1=axs[1].imshow(np.abs(AA), cmap='viridis')
axs[1].set_title('abs(A) Andreas')
plt.colorbar(im1, ax=axs[1])
# im2.set_clim(0,21)

# im2 = axs[2].imshow(np.abs(np.abs(A)-np.abs(AA)), cmap='viridis')
# axs[2].set_title('abs(abs(A)-abs(AA))')
# plt.colorbar(im0, ax=axs[2])

plt.show()


# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# im0 = axs[0].imshow(np.abs((A)-(AA)), cmap='viridis')
# axs[0].set_title('abs(A-AA)')
# plt.colorbar(im0, ax=axs[0])

# im1 = axs[1].imshow(np.real((A)-(AA)), cmap='viridis')
# axs[1].set_title('Re(A-AA)')
# plt.colorbar(im1, ax=axs[1])

# im2 = axs[2].imshow(np.imag((A)-(AA)), cmap='viridis')
# axs[2].set_title('Im(A-AA)')
# plt.colorbar(im1, ax=axs[2])


# How often do we disagree on values?
# print(f"How often do we disagree: {np.sum(np.abs(np.abs(A)-np.abs(AA)) > tol)}")
# print(f"Max difference: {max(abs((A)-(AA)).flatten())}")


# Comparison of RHS
fig, axs = plt.subplots(1, 1, figsize=(5, 5))

axs.plot(abs(b))
# axs.plot(abs(bA))
axs.set_title("Abs of RHS first polarization")
# axs.legend(["|b| (PN)", "|b| (Andreas)"])

# Comparisons of solutions
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# axs.plot(abs(abs(y)-abs(yA)))
axs.plot(abs(y))
axs.set_title("Abs of C first polarization")
