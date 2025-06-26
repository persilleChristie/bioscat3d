import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set tolerance for 0 values
tol = 1e-16


############# PN ############
# Load system matrix
df = pd.read_csv('../ForwardSolver/matrix_A_simple.csv', sep=",", header=None)

df = df.astype('string')

# Calculate condition number and number of zeros
A = df.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()
_, S, _ = np.linalg.svd(A)
print(f"Condition number PN: {S[0]/S[-1]}")

print(f"Total zeros: {np.sum(A<tol)}")
print(f"Actual zeros PN: {np.sum(A == 0)}")


# Load RHS
df1 = pd.read_csv('../ForwardSolver/vector_b_simple.csv', sep=",", header=None)

df1 = df1.astype('string')

b = df1.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()


# Load solution
df1 = pd.read_csv('../ForwardSolver/solution_y_simple.csv', sep=",", header=None)

df1 = df1.astype('string')

y = df1.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

############# ANDREAS #############
# Load system matrix
df1 = pd.read_csv('../ComparisonTest/System_comparison/A_Andreas.csv', sep=",", header=None, dtype=str)
# Convert to complex numbers
AA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

# Calculate condition number and number of zeros
_, S, _ = np.linalg.svd(AA)
print(f"Condition number A:  {S[0]/S[-1]}")
print(f"Total <<: {np.sum(AA<tol)}")
print(f"Actual zeros AA: {np.sum(AA == 0)}")

# Load solution
df1 = pd.read_csv('../ComparisonTest/System_comparison/y_Andreas.csv', sep=",", header=None, dtype=str)
yA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

# Load RHS
df1 = pd.read_csv('../ComparisonTest/System_comparison/b_Andreas.csv', sep=",", header=None, dtype=str)
bA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

############# COMPARISONS ##############
# How often do we agree on zeros?
print(f"Small values In same places?: {np.sum((AA < tol) * (A < tol))}")
print(f"Small values with big values?: {np.sum((AA < tol) * (A > tol) + (A < tol) * (AA > tol))}")
print(f"True zero In same places?: {np.sum((AA == 0) * (A == 0))}")

# Where do we disagree on zeros?
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
im0 = axs.imshow((A < tol) * (AA > tol), cmap='viridis')
axs.set_title("PN <<, Andreas not <<")
plt.colorbar(im0, ax=axs)

# Where do we disagree on True zeros?
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
im0 = axs.imshow((A == 0) * (AA != 0), cmap='viridis')
axs.set_title("PN True 0, Andreas not True 0")
plt.colorbar(im0, ax=axs)

# Where is A True zeros?
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
im0 = axs.imshow((A < tol), cmap='viridis')
axs.set_title("PN True 0")
plt.colorbar(im0, ax=axs)

print(f"Actual zeros: {np.sum(A < tol)}")

# Where is AA True zeros?
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
im0 = axs.imshow((AA == 0), cmap='viridis')
axs.set_title("AA True 0")
plt.colorbar(im0, ax=axs)

# Where do we agree?
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
im0 = axs.imshow((A < tol) * (AA < tol), cmap='viridis')
axs.set_title("PN 0, Andreas 0")
plt.colorbar(im0, ax=axs)


# Comparisons of absolute values
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

im0 = axs[0].imshow(np.abs(np.abs(A)-np.abs(AA)), cmap='viridis')
axs[0].set_title('abs(abs(A)-abs(AA))')
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(np.abs(A), cmap='viridis')
axs[1].set_title('abs(A)')
plt.colorbar(im1, ax=axs[1])
im1.set_clim(0,21)

im2=axs[2].imshow(np.abs(AA), cmap='viridis')
axs[2].set_title('abs(AA)')
plt.colorbar(im2, ax=axs[2])
im2.set_clim(0,21)

plt.show()


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im0 = axs[0].imshow(np.abs((A)-(AA)), cmap='viridis')
axs[0].set_title('abs(A-AA)')
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(np.real((A)-(AA)), cmap='viridis')
axs[1].set_title('Re(A-AA)')
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(np.imag((A)-(AA)), cmap='viridis')
axs[2].set_title('Im(A-AA)')
plt.colorbar(im1, ax=axs[2])


# How often do we disagree on values?
print(f"How often do we disagree: {np.sum(np.abs(np.abs(A)-np.abs(AA)) > tol)}")
print(f"Max difference: {max(abs((A)-(AA)).flatten())}")


# Comparison of RHS
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(abs(b))
axs.plot(abs(bA))
axs.set_title("Comparison of RHS")
axs.legend(["|b| (PN)", "|b| (Andreas)"])

# Comparisons of solutions
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(abs(abs(y)-abs(yA)))
axs.set_title("Difference in abs value of solutions")
