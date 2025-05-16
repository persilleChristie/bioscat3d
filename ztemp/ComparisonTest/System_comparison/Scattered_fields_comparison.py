import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########### PN #############
df = pd.read_csv('../../ForwardSolver/FilesCSV/scatteredFieldE_PN.csv', sep=",", header=None)
# Convert to complex numbers
df = df.astype('string')
E_PN = df.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

df = pd.read_csv('../../ForwardSolver/FilesCSV/scatteredFieldH_PN.csv', sep=",", header=None)
# Convert to complex numbers
df = df.astype('string')
H_PN = df.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()


########### ANDREAS #############
df1 = pd.read_csv('E_Andreas.csv', sep=",", header=None, dtype=str)
# Convert to complex numbers
E_A = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()

df1 = pd.read_csv('H_Andreas.csv', sep=",", header=None, dtype=str)
# Convert to complex numbers
H_A = df1.applymap(lambda x: complex(x.replace(' ', '')) if pd.notnull(x) else np.nan).to_numpy()


########### COMPARISON ###########
# Compute absolute difference for each column
differences = {}

differences["Ex_real"] = np.max(np.abs(np.real(E_A[:,0]) - np.real(E_PN[:,0])))
differences["Ey_real"] = np.max(np.abs(np.real(E_A[:,1]) - np.real(E_PN[:,1])))
differences["Ez_real"] = np.max(np.abs(np.real(E_A[:,2]) - np.real(E_PN[:,2])))
differences["Hx_real"] = np.max(np.abs(np.real(H_A[:,0]) - np.real(H_PN[:,0])))
differences["Hy_real"] = np.max(np.abs(np.real(H_A[:,1]) - np.real(H_PN[:,1])))
differences["Hz_real"] = np.max(np.abs(np.real(H_A[:,2]) - np.real(H_PN[:,2])))

differences["Ex_imag"] = np.max(np.abs(np.imag(E_A[:,0]) - np.imag(E_PN[:,0])))
differences["Ey_imag"] = np.max(np.abs(np.imag(E_A[:,1]) - np.imag(E_PN[:,1])))
differences["Ez_imag"] = np.max(np.abs(np.imag(E_A[:,2]) - np.imag(E_PN[:,2])))
differences["Hx_imag"] = np.max(np.abs(np.imag(H_A[:,0]) - np.imag(H_PN[:,0])))
differences["Hy_imag"] = np.max(np.abs(np.imag(H_A[:,1]) - np.imag(H_PN[:,1])))
differences["Hz_imag"] = np.max(np.abs(np.imag(H_A[:,2]) - np.imag(H_PN[:,2])))

differences["Ex_abs"] = np.max(np.abs(np.abs(E_A[:,0]) - np.abs(E_PN[:,0])))
differences["Ey_abs"] = np.max(np.abs(np.abs(E_A[:,1]) - np.abs(E_PN[:,1])))
differences["Ez_abs"] = np.max(np.abs(np.abs(E_A[:,2]) - np.abs(E_PN[:,2])))
differences["Hx_abs"] = np.max(np.abs(np.abs(H_A[:,0]) - np.abs(H_PN[:,0])))
differences["Hy_abs"] = np.max(np.abs(np.abs(H_A[:,1]) - np.abs(H_PN[:,1])))
differences["Hz_abs"] = np.max(np.abs(np.abs(H_A[:,2]) - np.abs(H_PN[:,2])))



relative_differences = {}

relative_differences["Ex_real"] = np.median(np.abs(np.real(E_A[:,0]) - np.real(E_PN[:,0]))/np.abs(np.real(E_PN[:,0])))
relative_differences["Ey_real"] = np.median(np.abs(np.real(E_A[:,1]) - np.real(E_PN[:,1]))/np.abs(np.real(E_PN[:,1])))
relative_differences["Ez_real"] = np.median(np.abs(np.real(E_A[:,2]) - np.real(E_PN[:,2]))/np.abs(np.real(E_PN[:,2])))
relative_differences["Hx_real"] = np.median(np.abs(np.real(H_A[:,0]) - np.real(H_PN[:,0]))/np.abs(np.real(H_PN[:,0])))
relative_differences["Hy_real"] = np.median(np.abs(np.real(H_A[:,1]) - np.real(H_PN[:,1]))/np.abs(np.real(H_PN[:,1])))
relative_differences["Hz_real"] = np.median(np.abs(np.real(H_A[:,2]) - np.real(H_PN[:,2]))/np.abs(np.real(H_PN[:,2])))

relative_differences["Ex_imag"] = np.median(np.abs(np.imag(E_A[:,0]) - np.imag(E_PN[:,0]))/np.abs(np.imag(E_PN[:,0])))
relative_differences["Ey_imag"] = np.median(np.abs(np.imag(E_A[:,1]) - np.imag(E_PN[:,1]))/np.abs(np.imag(E_PN[:,1])))
relative_differences["Ez_imag"] = np.median(np.abs(np.imag(E_A[:,2]) - np.imag(E_PN[:,2]))/np.abs(np.imag(E_PN[:,2])))
relative_differences["Hx_imag"] = np.median(np.abs(np.imag(H_A[:,0]) - np.imag(H_PN[:,0]))/np.abs(np.imag(H_PN[:,0])))
relative_differences["Hy_imag"] = np.median(np.abs(np.imag(H_A[:,1]) - np.imag(H_PN[:,1]))/np.abs(np.imag(H_PN[:,1])))
relative_differences["Hz_imag"] = np.median(np.abs(np.imag(H_A[:,2]) - np.imag(H_PN[:,2]))/np.abs(np.imag(H_PN[:,2])))

relative_differences["Ex_abs"] = np.median(np.abs(np.abs(E_A[:,0]) - np.abs(E_PN[:,0]))/np.abs(E_PN[:,0]))
relative_differences["Ey_abs"] = np.median(np.abs(np.abs(E_A[:,1]) - np.abs(E_PN[:,1]))/np.abs(E_PN[:,1]))
relative_differences["Ez_abs"] = np.median(np.abs(np.abs(E_A[:,2]) - np.abs(E_PN[:,2]))/np.abs(E_PN[:,2]))
relative_differences["Hx_abs"] = np.median(np.abs(np.abs(H_A[:,0]) - np.abs(H_PN[:,0]))/np.abs(H_PN[:,0]))
relative_differences["Hy_abs"] = np.median(np.abs(np.abs(H_A[:,1]) - np.abs(H_PN[:,1]))/np.abs(H_PN[:,1]))
relative_differences["Hz_abs"] = np.median(np.abs(np.abs(H_A[:,2]) - np.abs(H_PN[:,2]))/np.abs(H_PN[:,2]))


# Print differences
print("\nMax differences between Andreas and PN:")
for key, value in differences.items():
    print(f"{key}: {value}")

print("\nMedian relative differences between Andreas and PN:")
for key, value in relative_differences.items():
    print(f"{key}: {value}")


# Plots
