from laspy.file import File
import numpy as np
# import pptk
import open3d as o3d


def pre_processing(dataFile: File) -> np.ndarray:
    z_mean = dataFile.header.max[2] - dataFile.header.min[2]

    # print(dataFile.header.offset)
    # print(dataFile.header.max)
    # print(dataFile.header.min)

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

    def calculate_close_points(points, distances, max_distance):
        close = distances <= max_distance
        return points[close]

    def region_grow_plane(rand_point, close_plane_points: np.array):

        def point_str(point):
            return str(point[0]) + ',' + str(point[1]) + ',' + str(point[2])

        # region_points_dict = {}
        region_points = np.zeros((1,3))
        neighbour_points = np.array(rand_point)
        # all_neighbour_points_dict = {}
        # all_neighbour_points_dict[point_str(rand_point[0])] = rand_point[0]
        


        while neighbour_points.size != 0:
            # print('neighbour_points.shape:', neighbour_points.shape)
            # Take out the first point of the neighbour_points and add it to region_points
            neighbour = neighbour_points[0]
            neighbour_points = np.delete(neighbour_points, 0, axis=0)
            region_points = np.append(region_points, np.array([neighbour]), axis=0)

            # Calculate distances from the neighbour
            # TODO: Try different distances! Not just 0.5
            distances = np.sum((close_plane_points - np.array([neighbour]))**2, axis=1)
            close_to_neighbour = calculate_close_points(close_plane_points, distances, 1)

            # Add new neighbours to neighbour_points that are 
            # not already in neighbour_points or region_points
            for close in close_to_neighbour:
                if (not close in neighbour_points) and (not close in region_points):
                    neighbour_points = np.append(neighbour_points, np.array([close]), axis=0)
            

        # print('region_points:', region_points)
        for i in range(region_points.size):
            if (region_points[i] == np.zeros((1,3))).all():
                region_points = np.delete(region_points, i, axis=0)
                break
                
        # Loop through the points and add them to the region points if they are close to another open region point

        # region_points[str(rand_point[0]) + str(rand_point[1]) + str(rand_point[2])]
        return region_points

    # Find random point
    indx = np.random.randint(0, xyz.shape[0], 1)
    rand_point = xyz[indx]
    # print('rand_point:', rand_point)

    # Calculate distances from rand_point
    point_distances = np.sum((xyz - rand_point)**2, axis=1)

    # Only keep points that are close
    max_point_distance = 1.0  # 1m
    close_points = calculate_close_points(
        xyz, point_distances, max_point_distance)

    # If not enough close points, try again
    if close_points.shape[0] <= 1:
        print('No close points to rand_point')
        return RANSAC(xyz, iteration+1)

    # Find 2 other random points that are close to the rand_point, and that are not the same
    other_indxs = np.random.choice(close_points.shape[0], 2, replace=False)
    other_points = close_points[other_indxs]

    # Check if the plane has a good angle
    cp, d = plane_from_three_points(
        rand_point, other_points[0], other_points[1])

    # If the denominator is zero, start RANSAC again
    if np.abs(np.sum(cp**2, axis=1)) < 0.001:
        print('Denominator is zero')
        return RANSAC(xyz, iteration+1)

    # If the angle of the plane is not suitable, start RANSAC again
    # Suitable roof angles: between 15 and 60, over 90deg? remove 90 from it. negative? use abs.
    plane_angle = np.abs(calculate_plane_angle(cp))
    # if plane_angle > 90:
    #    plane_angle =

    # Check if plane angle is suitable for a roof, else start RANSAC again
    if 15 > plane_angle or plane_angle > 60:
        print('Bad plane_angle')
        return RANSAC(xyz, iteration+1)

    # print('Plane angle:', plane_angle)

    # Only use ponist close to the plane
    max_plane_distance = 0.1  # 10cm

    plane_distances = np.abs(np.sum(cp*xyz, axis=1) - d) / \
        np.sqrt(np.sum(cp**2, axis=1))

    close_plane_points = calculate_close_points(
        xyz, plane_distances, max_plane_distance)

    # Grow from the rand_point to the edge of the roof
    roof_points = region_grow_plane(rand_point, close_plane_points)
    
    if roof_points.shape[0] < 50:
        print('Too few points in roof_points:', roof_points.shape[0])
        return RANSAC(xyz, iteration+1)

    # print status about the points:
    print('------- DONE -------')
    print('Plane angle                        :', plane_angle)

    return roof_points, iteration, rand_point


# # # Read the data and preprocess it
dataFile = File('Data09.las', mode='r')

xyz = pre_processing(dataFile)


# # # Do RANSAC
RANSAC_points, iterations, rand_point = RANSAC(xyz, 0)

print('Total iterations in RANSAC         : ' + str(iterations))
print('Total number of points after RANSAC: ' + str(RANSAC_points.shape[0]))

# # # Show the points

# # pptk

# v = pptk.viewer(RANSAC_points, RANSAC_points[:, 2])
# v.set(point_size=0.05)
# v.color_map('jet', scale=[0, 5])

# v = pptk.viewer(xyz, xyz[:, 2])
# v.color_map('jet', scale=[0, 5])
# v.color_map([[0, 0, 0], [1, 1, 1]])

# v.load(RANSAC_points, RANSAC_points[:, 2])

# # open3d
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz)
cloud.paint_uniform_color([0.1, 0.1, 0.1])


# Elevate roof points to make them visible
RANSAC_points = RANSAC_points + [0, 0, 1]

roof = o3d.geometry.PointCloud()
roof.points = o3d.utility.Vector3dVector(RANSAC_points)
roof.paint_uniform_color([0.9, 0.1, 0.1])

r = o3d.geometry.PointCloud()
r.points = o3d.utility.Vector3dVector(rand_point + [0, 0, 2])
r.paint_uniform_color([0.1, 0.1, 0.9])

# , point_show_normal=True)
o3d.visualization.draw_geometries([cloud, roof, r])

# input()  # prevent end
