import numpy as np
from scipy import interpolate


def smooth_path_spline(rx, ry, smoothing_factor=0.5, resolution=0.05):
    """
    Smooth path using spline interpolation
    
    Inputs:
        rx, ry: Raw waypoints from A*
        smoothing_factor: Spline smoothing (0=exact fit, higher=more smooth)
        resolution: Step size for interpolation
    
    Outputs:
        smooth_x, smooth_y: Smoothed path
    """

    # Cannot interpolate a path with less than 2 points
    if len(rx) < 2:
        return rx, ry
    
    points = np.array(list(zip(rx, ry)))
    
    # Remove duplicate consecutive points 
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    points = points[np.sort(unique_indices)]
    
    # Re-check point count after duplicate removal
    if len(points) < 2:
        return list(points[:, 0]), list(points[:, 1])
    
    x = points[:, 0]
    y = points[:, 1]
    
    # B pline interpolation
    tck, u = interpolate.splprep([x, y], s=smoothing_factor, k=min(3, len(x)-1))
    
    # Calculate number of points based on total path length
    total_distance = 0
    for i in range(len(x)-1):
        total_distance += np.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
    
    num_points = max(int(total_distance / resolution), len(x) * 10)
    
    # Generate a uniform parametric vector from 0.0 to 1.0
    unew = np.linspace(0, 1.0, num_points)
    # Evaluate the spline at the new parametric points to get smooth coordinates
    out = interpolate.splev(unew, tck)
    
    return list(out[0]), list(out[1])
