from laspy.file import File
import numpy as np
import open3d as o3d


def pre_processing(dataFile: File) -> np.ndarray:
    # Coords of all the points
    coords = np.vstack((dataFile.x, dataFile.y, dataFile.z)).transpose()

    # Boolean array with value based on if the z-value is over z_threshold or not
    z_threshold = 105
    high = coords[:, 2] > z_threshold

    # Return only points where the z-value is over z_threshold
    return coords[high]


def RANSAC(xyz: np.ndarray, iteration: int):
    '''
    returns a points that are close to a randomly found plane
    '''
    def plane_from_three_points(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        cp = np.cross(v1, v2)
        # a, b, c = cp[0]
        d = np.dot(cp, p2)[0]
        return cp, d

    def calculate_plane_angle(cp):
        return np.degrees(np.arccos(cp[0][2]/np.sqrt(np.sum(cp**2))))

    def calculate_close_points(points, distances, max_distance, min_distance = -1):
        all_close = distances <= max_distance
        if min_distance > 0:
            inner = distances <= min_distance
            return points[np.logical_and(all_close, inner)]
        return points[all_close]

    def region_grow_plane(rand_point, close_plane_points: np.array):

        def point_str(point):
            return str(point[0]) + ',' + str(point[1]) + ',' + str(point[2])

        region_points = np.zeros((1,3))
        neighbour_points = np.array(rand_point)
        all_neighbour_points_dict = {}
        all_neighbour_points_dict[point_str(rand_point[0])] = True

        # Loop through the points and add them to the region points if they are close to another open region point
        while neighbour_points.shape[0] != 0:
            # Take out the first point of the neighbour_points and add it to region_points
            neighbour = neighbour_points[0]
            neighbour_points = np.delete(neighbour_points, 0, axis=0)
            region_points = np.append(region_points, np.array([neighbour]), axis=0)

            # Calculate distances from the neighbour
            # TODO: Try different distances! Not just 50cm
            max_neighbour_distance = 0.4 # 40cm
            distances = np.sqrt(np.sum((close_plane_points - np.array([neighbour]))**2, axis=1))
            close_to_neighbour = calculate_close_points(close_plane_points, distances, max_neighbour_distance)

            # Add new neighbours to neighbour_points that are 
            # not already in neighbour_points or region_points
            for close in close_to_neighbour:
                # if not (close in neighbour_points or close in region_points):
                if not point_str(close) in all_neighbour_points_dict:
                    neighbour_points = np.append(neighbour_points, np.array([close]), axis=0)
                    all_neighbour_points_dict[point_str(close)] = True

        # Remove the first point, as it is the zero-point
        region_points = np.delete(region_points, 0, axis=0)

        # region_points[str(rand_point[0]) + str(rand_point[1]) + str(rand_point[2])]
        return region_points

    # Find random point
    indx = np.random.randint(0, xyz.shape[0], 1)
    rand_point = xyz[indx]

    # Calculate distances from rand_point
    point_distances = np.sqrt(np.sum((xyz - rand_point)**2, axis=1))

    # Only keep points that are close but not too close
    max_point_distance = 1.5  # 1.5m
    min_point_distance = 1.0  # 1m
    close_points = calculate_close_points(xyz, point_distances, max_point_distance, min_point_distance)

    # If not enough close points, try again
    if close_points.shape[0] <= 1:
        print('No close points to rand_point')
        return RANSAC(xyz, iteration+1)

    # Find 2 other random points that are close to the rand_point, and that are not the same
    other_indxs = np.random.choice(close_points.shape[0], 2, replace=False)
    other_points = close_points[other_indxs]

    # Check if the plane has a good angle
    cp, _ = plane_from_three_points(
        rand_point, other_points[0], other_points[1])

    # If the denominator is zero, start RANSAC again
    if np.abs(np.sum(cp**2, axis=1)) < 0.0001:
        print('Denominator is zero')
        return RANSAC(xyz, iteration+1)

    # If the angle of the plane is not suitable, start RANSAC again
    # Suitable roof angles: between 15 and 60, over 90deg? remove 90 from it. negative? use abs.
    plane_angle = np.abs(calculate_plane_angle(cp))

    # Check if plane angle is suitable for a roof, else start RANSAC again
    if 15 > plane_angle or plane_angle > 60:
        print('Bad plane_angle')
        return RANSAC(xyz, iteration+1)

    # We do now know that we can construct a good plane from this rand_point
    # Extract a medium-large area
    max_area_distance = 40 # 40m

    # Use point_distancec which is defined from the rand_point above
    area_points = calculate_close_points(xyz, point_distances, max_area_distance)

    # Region grow from the rand_point on the area to only include the roof
    region_points = region_grow_plane(rand_point, area_points)

    # Check if the region has to few points, else start RANSAC again
    if region_points.shape[0] < 2000:
        print('Too few points in region:', region_points.shape[0])
        return RANSAC(xyz, iteration+1)


    # Class to store planes
    class Plane():
        def __init__(self, points: np.ndarray, angle: float):
            self.points = points
            self.angle = angle
        
        def __gt__(self, other):
            return self.points.shape[0] > other.points.shape[0]

    # Perform RANSAC on the region
    saved_ransac_roofs = np.array([Plane(np.zeros((1,3)), angle=0)], dtype=Plane)
    ransac_tot_runs = 50
    ransac_run_num = 0

    # TODO: Test different values
    max_plane_distance = 0.5  # 50cm

    
    while ransac_run_num < ransac_tot_runs:
        rand_indxs = np.random.choice(region_points.shape[0], 3, replace=False)
        rand_points = region_points[rand_indxs]

        ransac_cp, ransac_d = plane_from_three_points(np.array([rand_points[0]]), rand_points[1], rand_points[2])

        # If the denominator is zero, loop again
        if np.abs(np.sum(ransac_cp**2, axis=1)) < 0.0001:
            continue
        
        ransac_plane_angle = np.abs(calculate_plane_angle(ransac_cp))

        # Check if plane angle is not suitable for a roof, loop again
        if 15 > ransac_plane_angle or ransac_plane_angle > 60:
            continue

        # Calculate which points are in the plane
        ransac_plane_distances = np.abs(np.sum(ransac_cp*region_points, axis=1) - ransac_d) / np.sqrt(np.sum(ransac_cp**2, axis=1))
        ransac_plane_points = calculate_close_points(region_points, ransac_plane_distances, max_plane_distance)

        saved_ransac_roofs = np.append(saved_ransac_roofs, Plane(ransac_plane_points, ransac_plane_angle))

        ransac_run_num += 1


    # Coose the ransac_roof with the most points
    best_roof_index = np.argmax(saved_ransac_roofs)
    best_roof = saved_ransac_roofs[best_roof_index]
    
    if best_roof.points.shape[0] < 800:
        print('Too few points in best_roof:', best_roof.points.shape[0])
        return RANSAC(xyz, iteration+1)

    return best_roof.points, iteration # region_points, iteration #  


# # # Read the data and preprocess it
dataFile = File('Data09.las', mode='r')

xyz = pre_processing(dataFile)


# # # Do RANSAC several times and save the results
roofs = np.zeros((1,3))
for i in range(20):
    RANSAC_points, iterations = RANSAC(xyz, 0)
    print('------- DONE -------')
    print('Info about roof numer:', i + 1)
    print('Total iterations in RANSAC         : ' + str(iterations))
    print('Total number of points after RANSAC: ' + str(RANSAC_points.shape[0]))

    # Elevate roof points to make them visible
    RANSAC_points = RANSAC_points + [0, 0, 0.1]
    roofs = np.append(roofs, RANSAC_points, axis=0)

roofs = np.delete(roofs, 0, axis=0)
print(roofs.shape)

# # # Show the points
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz)
cloud.paint_uniform_color([0.1, 0.1, 0.1])

roof = o3d.geometry.PointCloud()
roof.points = o3d.utility.Vector3dVector(roofs)
roof.paint_uniform_color([0.9, 0.1, 0.1])


o3d.visualization.draw_geometries([cloud, roof])


# , point_show_normal=True)
