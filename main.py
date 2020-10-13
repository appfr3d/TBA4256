from laspy.file import File
import numpy as np
import pptk


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

    # Find random point
    indx = np.random.randint(0, xyz.shape[0], 1)
    rand_point = xyz[indx]

    # Calculate distances from rand_point
    point_distances = np.sum((xyz - rand_point)**2, axis=1)

    # Only keep points that are close
    max_point_distance = 1.0  # 1m
    close_points = calculate_close_points(
        xyz, point_distances, max_point_distance)

    # If not enough close points, try again
    if close_points.shape[0] <= 1:
        return RANSAC(xyz, iteration+1)

    # Find 2 other random points that are close to the rand_point, and that are not the same
    other_indxs = np.random.choice(close_points.shape[0], 2, replace=False)
    other_points = close_points[other_indxs]

    # Check if the plane has a good angle
    cp, d = plane_from_three_points(
        rand_point, other_points[0], other_points[1])

    # If the denominator is zero, start RANSAC again
    if np.abs(np.sum(cp**2, axis=1)) < 0.001:
        return RANSAC(xyz, iteration+1)

    # If the angle of the plane is not suitable, start RANSAC again
    # Suitable roof angles: between 15 and 60, over 90deg? remove 90 from it. negative? use abs.
    plane_angle = np.abs(calculate_plane_angle(cp))
    if plane_angle > 90:
        plane_angle -= 90

    # Check if plane angle is suitable for a roof, else start RANSAC again
    if 15 > plane_angle > 60:
        return RANSAC(xyz, iteration+1)

    print('Plane angle:', calculate_plane_angle(cp))

    max_plane_distance = 0.1  # 10cm

    plane_distances = np.abs(np.sum(cp*xyz, axis=1) - d) / \
        np.sqrt(np.sum(cp**2, axis=1))

    close_plane_points = calculate_close_points(
        xyz, plane_distances, max_plane_distance)

    # TODO: Make a region growing algorithm, and grow from the rand_point to the edge of the roof

    return close_plane_points, iteration


# # # Read the data and preprocess it
dataFile = File('Data09.las', mode='r')

xyz = pre_processing(dataFile)


# # # Do RANSAC
RANSAC_points, iterations = RANSAC(xyz, 0)

print('Total iterations in RANSAC: ' + str(iterations))
print('Total number of points after RANSAC: ' + str(RANSAC_points.shape[0]))

# # # Show the points

# v = pptk.viewer(RANSAC_points, RANSAC_points[:, 2])
# v.set(point_size=0.05)
# v.color_map('jet', scale=[0, 5])


v = pptk.viewer(xyz, xyz[:, 2])
# v.color_map('jet', scale=[0, 5])
# v.color_map([[0, 0, 0], [1, 1, 1]])

# v.load(RANSAC_points, RANSAC_points[:, 2])


input()  # prevent end
