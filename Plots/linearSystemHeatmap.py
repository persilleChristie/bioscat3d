import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tol = 1e-8

df = pd.read_csv('../ForwardSolver/matrix_A_simple.csv', sep=",", header=None)

df = df.astype('string')

A = df.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()
_, S, _ = np.linalg.svd(A)
print(f"Condition number PN: {S[0]/S[-1]}")

fig, axs = plt.subplots(1, 1, figsize=(5, 5))

im0 = axs.imshow(A < tol, cmap='viridis')
plt.colorbar(im0, ax=axs)

print(f"Total zeros: {np.sum(A<tol)}")


df1 = pd.read_csv('../ComparisonTest/System_comparison/A_Andreas.csv', sep=",", header=None, dtype=str)
# Convert to complex numbers
AA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

_, S, _ = np.linalg.svd(AA)
print(f"Condition number A:  {S[0]/S[-1]}")
print(f"Total zeros: {np.sum(AA<tol)}")

print(f"In same places?: {np.sum((AA<tol) * (A < tol))}")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

im0 = axs[0].imshow(abs(abs(A)-abs(AA)), cmap='viridis')
axs[0].set_title('abs(abs(A)-abs(AA))')
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(abs(A), cmap='viridis')
axs[1].set_title('abs(A)')
plt.colorbar(im1, ax=axs[1])

im2=axs[2].imshow(abs(AA), cmap='viridis')
axs[2].set_title('abs(AA)')
plt.colorbar(im2, ax=axs[2])

plt.show()

print(f"How often do we disagree: {np.sum(abs(abs(A)-abs(AA)) < tol)}")
print(f"Max difference: {max(abs(A-AA).flatten())}")

df1 = pd.read_csv('../ForwardSolver/solution_y_simple.csv', sep=",", header=None)

df1 = df1.astype('string')

y = df1.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()


df1 = pd.read_csv('../ComparisonTest/System_comparison/y_Andreas.csv', sep=",", header=None, dtype=str)
# Convert to complex numbers
yA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

fig, axs = plt.subplots(1, 1, figsize=(5, 5))

axs.plot(abs(abs(y)-abs(yA)))


df1 = pd.read_csv('../ForwardSolver/vector_b_simple.csv', sep=",", header=None)

df1 = df1.astype('string')

b = df1.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()


df1 = pd.read_csv('../ComparisonTest/System_comparison/b_Andreas.csv', sep=",", header=None, dtype=str)
# Convert to complex numbers
bA = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

fig, axs = plt.subplots(1, 1, figsize=(5, 5))

axs.plot(abs(b))
axs.plot(abs(bA))
