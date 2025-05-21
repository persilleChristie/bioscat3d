import numpy as np
import pandas as pd
from Spline import Spline

def save_to_csv(points, normals, tangent1, tangent2, filename):
    df = pd.DataFrame({"x": points[:,0], "y": points[:,1], "z": points[:,2], 
                       "nx": normals[:,0], "ny": normals[:,1], "nz": normals[:,2],
                       "t1x": tangent1[:,0], "t1y": tangent1[:,1], "t1z": tangent1[:,2],
                       "t2x": tangent2[:,0], "t2y": tangent2[:,1], "t2z": tangent2[:,2]})
    df.to_csv(filename, index = False)


lambdas = np.linspace(0.325, 1, 2)
lambda_min = lambdas.min()
a, b = [-1, 1]

bump = lambda x,y,x0,y0,height,sigma: height*np.exp(
        -( (x-x0)**2 + (y-y0)**2 ) / (2*sigma**2)
    )
f = lambda x,y: (
                    bump(x,y,-0.250919762305275, 0.9014286128198323, 0.24639878836228102, 0.49932924209851826)+
                    bump(x,y,-0.687962719115127, -0.6880109593275947, 0.1116167224336399, 0.6330880728874675) +
                    bump(x,y,0.2022300234864176, 0.416145155592091, 0.10411689885916049,0.6849549260809971) 
                )

max_resolution = int(np.ceil(np.sqrt(2) * 5 * (b-a)/lambda_min))
xfine = np.linspace(a, b, max_resolution)
yfine = np.linspace(a, b, max_resolution)
Xfine, Yfine = np.meshgrid(xfine, yfine)
Zfine = f(Xfine, Yfine)

spline = Spline(Xfine, Yfine, Zfine)

for i, lbda in enumerate(lambdas):
    scale = int(np.ceil((b-a)/lbda))
    resol_test = int(np.ceil(np.sqrt(2) * 5 * scale))
    testpoints, normals, tangent1, tangent2 = spline.calculate_points(resol_test)
    
    resol_ref = 10 * scale
    refpoints, ref_normals, ref_tangent1, ref_tangent2 = spline.calculate_points(resol_ref)

    save_to_csv(testpoints, normals, tangent1, tangent2, "Python/Splines/FilesCSV/testpoints_"+str(i)+".csv")
    save_to_csv(refpoints, ref_normals, ref_tangent1, ref_tangent2, "Python/Splines/FilesCSV/refpoints_"+str(i)+".csv")

    

    

