from laspy.file import File
import numpy as np
import open3d as o3d
import pptk


class RANSAC():
  def __init__(self):
    # Preprocess
    # Read data from file
    dataFile = File('trees_no_ground.las', mode='r')
    self.max = dataFile.header.max
    self.min = dataFile.header.min
    print('Max z:', self.max[2], ', min z:', self.min[2])
    # Max z: 77.28919982910156 , min z: -25.123899459838867
    self.xyz = np.vstack((dataFile.x, dataFile.y, dataFile.z)).transpose()
    self.sorted_cloud = None

  def run_calculations(self):
    # Terrain is already removed by using Cloud compare with the package CSF filter

    # Sort cloud and store for later
    self.sorted_cloud = self.sort_cloud(self.xyz)

    # Slice point cloud in horizontal direction
    num_slices = 10
    slices = self.slice_cloud(self.sorted_cloud, num_slices)

    # Find a good part of the cloud to do RANSAC on
    good_part = self.find_good_part(slices[4])



    # RANSAC on circles in a slice in 2D by disregarding the z-value
    three = self.run_ransac(slices[4])

    # Detect the "same" circles in at least 3 layers

    # Set op a 3D bounding box around the tree stems

    # Export the point clouds inside the bounding boxes
    return three

  def sort_cloud(self, cloud):
    return cloud[np.argsort(cloud[:, 2])]
    # return np.sort(cloud, axis=0)

  def slice_cloud(self, cloud, num_slices):
    return np.array_split(cloud, num_slices)

  def find_good_part(self, cloud):
    # Find a random initial point
    indx = np.random.randint(0, cloud.shape[0], 1)
    rand_point = cloud[indx]
    print(cloud.shape)
    print(rand_point.shape)

    # Find the distnces to all other points in the cloud
    # Only take the distance on x and y, not z
    point_distances = np.sqrt(np.sum((cloud[:,:2] - rand_point[:2])**2, axis=1))

    # Find close points to the random point
    max_point_distance = 1 # 1 distance (Do not know the scale of this number yet.)
    close_points = self.calculate_close_points(cloud, point_distances, max_point_distance)

    # There must be a minimum of X points in the close_points
    if close_points.chape[0] < 100:
      return self.find_good_part(cloud)

    # Choose two new random points that are close to the rand_point, and are not the same
    other_indxs = np.random.choice(close_points.shape[0], 2, replace=False)
    other_points = close_points[other_indxs]

    # Make a circle with our three random points
    # TODO: make a function that calculates circle paameters from three points

    # A tree should be a circle with many points in the circle, but few or none points outside the circle



    print(point_distances.shape)

    return cloud
  
  def calculate_close_points(self, points, distances, max_distance, min_distance = -1):
    '''
      Find all points in points that have a maximum ditance max_distance (and minimum distance min_distance)
    '''
    all_close = distances <= max_distance
    if min_distance > 0:
        inner = distances <= min_distance
        return points[np.logical_and(all_close, inner)]
    return points[all_close]

  def run_ransac(self, cloud):
    # TODO: Option to pass in circle parameters for when we need to find the same circles in other layers
    # Choose a random point
    indx = np.random.randint(0, cloud.shape[0], 1)
    rand_point = cloud[indx]

    print(cloud.shape)
    print(rand_point.shape)

    # Only take the distance on x and y, not z
    point_distances = np.sqrt(np.sum((cloud[:,:2] - rand_point[:2])**2, axis=1))
    print(point_distances.shape)

    return cloud


if __name__ == "__main__":
  # Initialice the class
  ransac = RANSAC()

  # Run the calculations
  threes = ransac.run_calculations()

  # Generate o3d cloud
  # cloud = o3d.geometry.PointCloud()
  # cloud.points = o3d.utility.Vector3dVector(threes)
  # cloud.paint_uniform_color([0.1, 0.1, 0.1])

  # Visualize the cloud
  # o3d.visualization.draw_geometries([cloud])

  # Visualize with pptk
  v = pptk.viewer(threes, threes[:, 2])

  #v.color_map('summer', scale=None)
  v.color_map('gray')
  v.set(point_size=0.01)




