import numpy as np
import matplotlib.pyplot as plt
import math
import utm
import pymap3d as pm
import pandas as pd
#from keplergl import KeplerGl

#Depending on the precision needed later on check np.float128 or np.longdouble 


def parse_nmea(sentence):
    parts = sentence.split(',')
    # parts[2] = latitude in ddmm.mmmm, parts[3] = N/S
    # parts[4] = longitude in dddmm.mmmm, parts[5] = E/W
    # parts[9] = altitude in meters
    lat_str = parts[2]
    lat_dir = parts[3]
    lon_str = parts[4]
    lon_dir = parts[5]
    alt_str = parts[9]
    lat_deg = float(lat_str[:2])
    lat_min = float(lat_str[2:])
    time_str = parts[1]
    time_seconds = float(float(time_str[:2])*3600 + float(time_str[2:4])*60 + float(time_str[4:]))

    lat = lat_deg+(lat_min/60)
    if lat_dir == 'S':
        lat = -1*lat

    lon_deg = float(lon_str[:3])
    lon_min = float(lon_str[3:])
    lon = lon_deg + (lon_min/60)
    if lon_dir == 'W':
       lon = -1*lon
    
    alt = float(alt_str)
    return lat,lon,alt, time_seconds


# Cumulative Error Distribution (ECDF) Plots

def plot_ecdf(data, label):
    """
    Helper function to plot the empirical cumulative distribution function (ECDF).
    """
    sorted_data = np.sort(data)
    # Create cumulative probability values from 0 to 1.
    cum_prob = np.linspace(0, 1, len(sorted_data))
    plt.plot(sorted_data, cum_prob, label=label, linewidth=2)

if __name__ == '__main__':
    # Sample NMEA sentences (GPGGA format)
    gps_sentences = [
        "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47",
        "$GPGGA,123520,4807.045,N,01131.005,E,1,08,0.9,545.5,M,46.9,M,,*48",
        "$GPGGA,123521,4807.052,N,01131.010,E,1,08,0.9,545.6,M,46.9,M,,*49",
        "$GPGGA,123522,4807.060,N,01131.015,E,1,08,0.9,545.7,M,46.9,M,,*50",
        "$GPGGA,123523,4807.068,N,01131.020,E,1,08,0.9,545.8,M,46.9,M,,*51",
        "$GPGGA,123524,4807.075,N,01131.025,E,1,08,0.9,545.9,M,46.9,M,,*52",
        "$GPGGA,123525,4807.082,N,01131.030,E,1,08,0.9,546.0,M,46.9,M,,*53",
        "$GPGGA,123526,4807.090,N,01131.035,E,1,08,0.9,546.1,M,46.9,M,,*54",
        "$GPGGA,123527,4807.098,N,01131.040,E,1,08,0.9,546.2,M,46.9,M,,*55",
        "$GPGGA,123528,4807.105,N,01131.045,E,1,08,0.9,546.3,M,46.9,M,,*56"
    ]

    # Corresponding ground truth data (latitude, longitude in decimal degrees, altitude in meters)
    ground_truth = [
        (48.1173, 11.51667, 545.0),
        (48.1174, 11.51680, 545.3),
        (48.1175, 11.51690, 545.5),
        (48.1177, 11.51690, 545.8),
        (48.1178, 11.51700, 545.9),
        (48.1179, 11.51710, 545.8),
        (48.1180, 11.51720, 546.0),
        (48.1181, 11.51730, 546.1),
        (48.1183, 11.51730, 546.2),
        (48.1184, 11.51740, 546.3)
    ]

    #parse the GPS sentences
    parsed_data = [parse_nmea(sentence) for sentence in gps_sentences]
    parsed_data = np.array(parsed_data)
    parsed_data = parsed_data.astype(np.float64)
    gt_data = np.array(ground_truth)
    time_data = parsed_data[:,3]
    time_data = np.array(time_data)

    #Compute the errors (parse - ground truth)
    errors = parsed_data[:,:3] - gt_data
    lat_errors = errors[:,0]
    lon_errors = errors[:,1]
    alt_errors = errors[:,2]

    utm_east = []
    utm_north = []
    for point in parsed_data:
        easting, northing, zone_number, zone_letter = utm.from_latlon(point[0],point[1])
        utm_east.append(easting)
        utm_north.append(northing)

    gt_utm_east = []
    gt_utm_north = []
    for gt_point in gt_data:
        easting, northing, zone_number, zone_letter = utm.from_latlon(gt_point[0],gt_point[1])
        gt_utm_east.append(easting)
        gt_utm_north.append(northing)

    utm_east = np.array(utm_east)
    gt_utm_east = np.array(gt_utm_east)
    utm_north = np.array(utm_north)
    gt_utm_north = np.array(gt_utm_north)


    utm_east_errors = utm_east-gt_utm_east
    utm_north_errors = utm_north - gt_utm_north
    rms_utm_east_errors = np.sqrt(np.mean(utm_east_errors**2))
    rms_utm_north_errors = np.sqrt(np.mean(utm_north_errors**2))
    rms_alt_errors = np.sqrt(np.mean(alt_errors**2))

    print("RMS Errors\n","\nRMS Easting UTM\n", rms_utm_east_errors, "\nRMS Northing UTM\n",rms_utm_north_errors,"\nRMS Altitude Errors\n",rms_alt_errors)

    lat0, lon0, alt0 = gt_data[0,0], gt_data[0,1], gt_data[0,2]

    enu_e = []
    enu_n = []
    enu_u = []

    gt_enu_e = []
    gt_enu_n = []
    gt_enu_u = []

    # Convert geodetic to ENU coordinates
    for point in parsed_data:
        e, n, u = pm.geodetic2enu(point[0], point[1], point[2], lat0, lon0, alt0)
        enu_e.append(e)
        enu_n.append(n)
        enu_u.append(u)

    for gt_point in gt_data:
        e, n, u = pm.geodetic2enu(gt_point[0], gt_point[1], gt_point[2], lat0, lon0, alt0)
        gt_enu_e.append(e)
        gt_enu_n.append(n)
        gt_enu_u.append(u)

    enu_e = np.array(enu_e)
    enu_n = np.array(enu_n)
    enu_u = np.array(enu_u)

    gt_enu_e = np.array(gt_enu_e)
    gt_enu_n = np.array(gt_enu_n)
    gt_enu_u = np.array(gt_enu_u)

    errors_enu_e = np.array(enu_e-gt_enu_e)
    errors_enu_n = np.array(enu_n-gt_enu_n)
    errors_enu_u = np.array(enu_u-gt_enu_u)

    ecef_x = []
    ecef_y = []
    ecef_z = []
    
    # Convert from geodetic to ECEF
    for point in parsed_data:
        x, y, z = pm.geodetic2ecef(point[0], point[1], point[2])
        ecef_x.append(x)
        ecef_y.append(y)
        ecef_z.append(z)
    
    gt_ecef_x =[]
    gt_ecef_y = []
    gt_ecef_z = []

    for gt_point in gt_data:
        x,y,z = pm.geodetic2ecef(gt_point[0],gt_point[1],gt_point[2])
        gt_ecef_x.append(x)
        gt_ecef_y.append(y)
        gt_ecef_z.append(z)
    
    errors_ecef_x = np.array(ecef_x)-np.array(gt_ecef_x)
    errors_ecef_y = np.array(ecef_y)-np.array(gt_ecef_y)
    errors_ecef_z = np.array(ecef_z)-np.array(gt_ecef_z)

    # Plotting the errors in UTM and ENU frames
    
    print("PARSED DATA\n", parsed_data)
    plt.figure()

    plt.subplot(4,3,1)
    plt.plot(time_data,utm_east_errors,marker = 'o',label='Easting Error')
    plt.plot(time_data,utm_north_errors,marker='s',label='Northing Error')
    plt.plot(time_data,alt_errors,marker='^',label='Altitude Error')
    plt.xlabel('Time (seconds since midnight)')
    plt.ylabel('Error (m)')
    plt.title('UTM Frame Positioning Errors vs Time')
    plt.legend()

    plt.subplot(4,3,2)
    plt.plot(time_data,errors_enu_e,marker = 'o',label='East Error')
    plt.plot(time_data,errors_enu_n,marker='s',label='North Error')
    plt.plot(time_data,errors_enu_u,marker='^',label='Up Error')
    plt.xlabel('Time (seconds since midnight)')
    plt.ylabel('Error (m)')
    plt.title('ENU Frame Positioning Errors vs Time')
    plt.legend()

    plt.subplot(4, 3, 3)
    plt.hist(utm_east_errors, bins=20, edgecolor='black')
    plt.xlabel("UTM Easting Error (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of UTM Easting Errors")

    plt.subplot(4, 3, 4)
    plt.hist(utm_north_errors, bins=20, edgecolor='black')
    plt.xlabel("UTM Northing Error (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of UTM Northing Errors")

    plt.subplot(4, 3, 5)
    plt.hist(alt_errors, bins=20, edgecolor='black')
    plt.xlabel("Altitude Error (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Altitude Errors")

    # Create a list of error arrays for UTM errors: Easting, Northing, and Altitude
    plt.subplot(4,3,6)
    utm_data = [utm_east_errors, utm_north_errors, alt_errors]
    plt.boxplot(utm_data, labels=['UTM Easting Error', 'UTM Northing Error', 'Altitude Error'])
    plt.title("Boxplot of UTM Errors (with Quartile Ranges)")
    plt.ylabel("Error (m)")
    plt.grid(True, linestyle='--', alpha=0.5)
    

    plt.subplot(4,3,7)
    enu_data = [errors_enu_e, errors_enu_n, alt_errors]
    plt.boxplot(enu_data, labels=['ENU Easting Error', 'ENU Northing Error', 'Altitude Error'])
    plt.title("Boxplot of UTM Errors (with Quartile Ranges)")
    plt.ylabel("Error (m)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # CDF for UTM Errors
    plt.subplot(4,3,8)
    plot_ecdf(abs(utm_east_errors), "UTM Easting Error")
    plot_ecdf(abs(utm_north_errors), "UTM Northing Error")
    plot_ecdf(abs(alt_errors), "Altitude Error")
    plt.xlabel("Error (m)")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of UTM Errors")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # percentiles for UTM errors

    # Define percentiles to compute from 0 to 100 (inclusive)
    percentiles = np.linspace(0, 100, 101)

    # Compute the percentiles for each error metric (absolute values)
    utm_e_pct = np.percentile(np.abs(utm_east_errors), percentiles)
    utm_n_pct = np.percentile(np.abs(utm_north_errors), percentiles)
    alt_pct   = np.percentile(np.abs(alt_errors), percentiles)

    # Plot the percentile graph for UTM errors
    plt.subplot(4,3,9)
    plt.plot(utm_e_pct,percentiles, label='UTM Easting Error', linewidth=2)
    plt.plot(utm_n_pct,percentiles, label='UTM Northing Error', linewidth=2)
    plt.plot(alt_pct,percentiles,   label='Altitude Error', linewidth=2)
    plt.ylabel('Percentile (%)')
    plt.xlabel('Error (m)')
    plt.title('Percentile Graph for UTM Errors')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    

    # Compute the 95th percentile errors for UTM axes
    
    utm_e_95 = np.percentile(np.abs(utm_east_errors), 95)
    utm_n_95 = np.percentile(np.abs(utm_north_errors), 95)
    alt_95   = np.percentile(np.abs(alt_errors), 95)

    print("95th Percentile UTM Errors:")
    print("UTM Easting Error:  {:.4f} m".format(utm_e_95))
    print("UTM Northing Error: {:.4f} m".format(utm_n_95))
    print("Altitude Error:     {:.4f} m".format(alt_95))

    plt.subplot(4,3,10)
    plt.plot(time_data,errors_ecef_x,marker='o',label= "ECEF X Error")
    plt.plot(time_data,errors_ecef_y,marker='s',label= "ECEF Y Error")
    plt.plot(time_data,errors_ecef_z,marker='^',label= "ECEF Z Error")
    plt.title("ECEF Frame Errors (m) Vs Time (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.5)


    

    q1, q3 = np.percentile(np.abs(utm_east_errors), [25, 75])
    iqr = q3 - q1

    # Define the lower and upper bounds for non-outlier data
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr    

    q1n,q3n = np.percentile(np.abs(utm_north_errors),[25,75])
    iqrn = q3n-q1n
    lb_n = q1n - 1.5*iqrn
    ub_n = q3n+1.5*iqrn

    q1a,q3a = np.percentile(np.abs(alt_errors),[25,75])
    iqra = q3a-q1a
    lb_a = q1a-1.5*iqra
    ub_a = q3a+1.5*iqra

    utm_east_filt_error = []
    utm_north_filt_error = []
    utm_alt_filt_error = []
    time_filt = []

    for i in range(0,len(utm_east_errors)):
        if (utm_east_errors[i] < lower_bound or utm_east_errors[i] > upper_bound or utm_north_errors[i] < lb_n or utm_north_errors[i]>ub_n or alt_errors[i]<lb_a or alt_errors[i]>ub_a):
            continue
        else:
            utm_east_filt_error.append(utm_east_errors[i])
            utm_north_filt_error.append(utm_north_errors[i])
            utm_alt_filt_error.append(alt_errors[i])
            time_filt.append(time_data[i])
    
    utm_east_filt_error = np.array(utm_east_filt_error)
    utm_north_filt_error = np.array(utm_north_filt_error)
    utm_alt_filt_error = np.array(utm_alt_filt_error)    
    
    plt.subplot(4,3,11)
    utm_data= [utm_east_filt_error,utm_north_filt_error,utm_alt_filt_error]
    plt.boxplot(utm_data, labels=['UTM Easting Error', 'UTM Northing Error', 'Altitude Error'])
    plt.title("Boxplot of UTM Filtered Errors with Quartile Range")
    plt.ylabel("Error (m)")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


"""
    # using example data for the predicted trajectory (with timestamp, lat, lon, alt)
    data_pred = pd.DataFrame({
        'latitude': [37.7749, 37.7750, 37.7751, 37.7752],
        'longitude': [-122.4194, -122.4193, -122.4192, -122.4191],
        'altitude': [10, 20, 30, 40],
    })

    # using example data for the ground truth trajectory
    data_gt = pd.DataFrame({
        'latitude': [37.7748, 37.7749, 37.7750, 37.7751],
        'longitude': [-122.4195, -122.4194, -122.4193, -122.4192],
        'altitude': [12, 22, 32, 42],
    })


    # Create a Kepler.gl map instance with a specified height
    map_ = KeplerGl(height=600)

    # Add the predicted trajectory data to the map
    map_.add_data(data=data_pred, name="Predicted Trajectory")

    # Add the ground truth data to the map
    map_.add_data(data=data_gt, name="Ground Truth Trajectory")

    # Save the map to an HTML file for sharing/viewing
    map_.save_to_html(file_name="kepler_map.html")""
"""
