import numpy as np
import pandas as pd
#Getting the sigma of User range error from a set of gps points, their corresponding dop values and the known static ground truth, a couple different methods have been used

# Set a random seed for reproducibility
np.random.seed(0)

# Assume the true sigma URE (in meters) for simulation
true_sigma_ure = 2.0

# Number of simulated measurements
n = 200  # (In a real scenario, this would be many more, over 12 hours)

# Simulate random HDOP and VDOP values
hdop = np.random.uniform(0.8, 1.5, n)  # Horizontal DOP values
vdop = np.random.uniform(1.0, 2.0, n)  # Vertical DOP values

# Ground truth location (for simplicity, assume at the origin)
gt_x, gt_y, gt_z = 0, 0, 0

# Generate measured positions:
# For horizontal measurements, the errors in x and y are drawn from a Gaussian distribution 
# with standard deviation = true_sigma_ure * HDOP.
err_x = np.random.normal(0, true_sigma_ure * hdop)
err_y = np.random.normal(0, true_sigma_ure * hdop)
# For the vertical measurement, the error is drawn with std = true_sigma_ure * VDOP.
err_z = np.random.normal(0, true_sigma_ure * vdop)

# Measured positions
measured_x = gt_x + err_x
measured_y = gt_y + err_y
measured_z = gt_z + err_z

horizontal_error = np.sqrt((measured_x-gt_x)**2 + (measured_y-gt_y)**2)
vertical_error = abs(measured_z-gt_z)
#print("\nHorizontal Error\n", horizontal_error," \nHorizontal Sig ",horizontal_sig,"\nHDOP\n", hdop)
horizontal_sig = horizontal_error/hdop
print("\nHorizontal Error\n", horizontal_error," \nHorizontal Sig ",horizontal_sig,"\nHDOP\n", hdop)
vertical_sig = vertical_error/vdop
error_sig = np.vstack((horizontal_sig,vertical_sig))
error_sig_ure = np.percentile(error_sig, 68)

print("\nError Sigma URE 1 sigma\n",error_sig_ure)

#Other approach - Weighted Least Squares
sigma_ure_els = (np.sum((horizontal_error*hdop))+np.sum((vertical_error*vdop)))/((hdop@(hdop.T))+(vdop@(vdop.T)))
print("Sigma URE WLS",sigma_ure_els)

#Sigma URE Mean 
sigma_ure_mean = np.mean(np.vstack((horizontal_error/hdop,vertical_error/vdop)))
print("Sigma URE Mean",sigma_ure_mean)
