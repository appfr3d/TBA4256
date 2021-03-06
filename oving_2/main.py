from laspy.file import File
import numpy as np
import open3d as o3d
# import pptk

# Taken from: https://stackoverflow.com/a/50974391/3692511
def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

class Circle():
  def __init__(self, center, radius, points):
    self.center = center
    self.radius = radius
    self.points = points

  def __gt__(self, other):
    return self.points.shape[0] > other.points.shape[0]


class RANSAC():
  def __init__(self):
    # Preprocess
    # Read data from file
    dataFile = File('original_slice_new.las', mode='r')
    self.max = dataFile.header.max
    self.min = dataFile.header.min
    # print('Max z:', self.max[2], ', min z:', self.min[2])
    # Max z: 3.0826001167297363 , min z: 0.1096000000834465
    # Max z: 77.28919982910156 , min z: -25.123899459838867
    self.xyz = np.vstack((dataFile.x, dataFile.y, dataFile.z)).transpose()
    self.sorted_cloud = None

  def run_calculations(self):
    # Terrain is already removed by using Cloud compare with the package CSF filter
    # Sort cloud and store for later
    self.sorted_cloud = self.sort_cloud(self.xyz)

    # Slice point cloud in horizontal direction
    num_slices = 20
    start_slice = 0
    slices = self.slice_cloud(self.sorted_cloud, num_slices)

    tree = None
    diameter = 0
    tree_found = False
    max_point_distance = 0.5 # Do not know the scale of this number yet

    while not tree_found:
      # Find a good part of the cloud to do RANSAC on
      good_part, center = self.find_good_part(slices[start_slice], max_point_distance)

      # RANSAC on circles in a slice in 2D by disregarding the z-value
      tree_part, variance = self.run_ransac(good_part)

      # If the centers have a too high variance, try again
      if variance > 1:
        print("tree part 0 has to high variance:", variance)
        continue

      # If the circle is not good, try again
      if tree_part.points.shape[0] < good_part.shape[0] - tree_part.points.shape[0]:
        print("tree part 0 is not good...")
        continue
      
      # Detect the "same" circles in at least two more layers
      good = True

      # The range could be increased to check more layers
      # If increased, the logic for when to break should be changed, as if 9/10 layers are good, 
      # it would still not consider it as a good tree
      for i in range(start_slice + 1, start_slice + 3):
        # Calculate point distances to the center in the next layer
        point_distances = self.calculate_circle_ditances(slices[i], center)

        # Keep the close points
        # TODO: Manye increase the max, dinstance, since the tree can be skewed and not strictly vertical
        close_points = self.calculate_close_points(slices[i], point_distances, max_point_distance)

        # If no points in this layer then start again
        if close_points.shape[0] <= 50:
          print("tree part", i, "has to few points...")
          good = False
          break

        # Do ransac
        next_tree_part, next_variance = self.run_ransac(close_points)

        # If the centers have a too high variance, try again
        if next_variance > 1:
          print("tree part", i, "has to high variance:", next_variance)
          continue
        
        # Old code, not removed to remeber it in the paper
        #if np.sqrt(np.sum((np.array(tree_part.center) - np.array(next_tree_part.center))**2)) > 0.15:
          #print("tree part", i, "has center to far appart from tree part 0...")
          #good = False
          #break

        # If the circle is not good in this layer, try again
        if next_tree_part.points.shape[0] < close_points.shape[0] - next_tree_part.points.shape[0]:
          print("tree part", i, "is not good...")
          good = False
          break
      
      if not good:
        continue

      # Else, good! Make the tree
      # Set up a 3D bounding box around the tree stems
      all_distances = self.calculate_circle_ditances(self.sorted_cloud, center)
      tree = self.calculate_close_points(self.sorted_cloud, all_distances, max_point_distance + max_point_distance * 0.8)
      # diameter = tree_part.radius * 2
      tree_found = True

    # Export the point clouds inside the bounding boxes
    return tree, tree_part

  def sort_cloud(self, cloud):
    return cloud[np.argsort(cloud[:, 2])]

  def slice_cloud(self, cloud, num_slices):
    return np.array_split(cloud, num_slices)

  def find_good_part(self, cloud, max_point_distance):
    # Find a random initial point
    indx = np.random.randint(0, cloud.shape[0], 1)
    rand_point = cloud[indx]

    # Find the distnces to all other points in the cloud
    # Only take the distance on x and y, not z
    point_distances = np.sqrt(np.sum((cloud[:,:2] - rand_point[:,:2])**2, axis=1))

    # Find close points to the random point
    close_points = self.calculate_close_points(cloud, point_distances, max_point_distance)

    # There must be a minimum of X points in the close_points
    if close_points.shape[0] < 1000:
      return self.find_good_part(cloud, max_point_distance)

    # Calculate the center of the points
    center = np.mean(close_points[:,:2], axis=0)

    # A tree should be a circle with many points in the circle, but few or none points outside the circle
    center_distances = self.calculate_circle_ditances(close_points, center)
    max_circle_distance = np.max(center_distances)

    circle_points = self.calculate_close_points(close_points, center_distances, max_circle_distance, max_circle_distance*0.3)
    # If there are more points in the middle than in the outer part, try again
    if circle_points.shape[0] < close_points.shape[0] - circle_points.shape[0]:
      print("Too many points in the center...")
      return self.find_good_part(cloud, max_point_distance)

    # Some part of an actual tree can be removed in circle_points
    # Therefore we must return the whole set of close_points
    return close_points, center
  
  def calculate_circle_ditances(self, points, center):
    # Only take the distance on x and y, not z
    return np.sqrt(np.sum((points[:,:2] - np.array(center))**2, axis=1))

  def calculate_close_points(self, points, distances, max_distance, min_distance = -1):
    '''
      Find all points in points that have a maximum ditance max_distance (and minimum distance min_distance)
    '''
    all_close = distances <= max_distance
    if min_distance > 0:
        inner = distances >= min_distance
        return points[np.logical_and(all_close, inner)]
    return points[all_close]

  def run_ransac(self, cloud):
    num_runs = 20

    # Save circles, add a mock Cricle first to get a good shape
    saved_ransac_circles = np.array([Circle([0,0], 0, np.zeros((1,3)))], dtype=Circle)

    while saved_ransac_circles.shape[0] < num_runs:
      # TODO: just choose 3 points at once, instead of 1 and then 2
      # Choose a random point
      # indx = np.random.randint(0, cloud.shape[0], 1)
      # rand_point = cloud[indx]

      # Choose three random points that are not the same
      rand_indices = np.random.choice(cloud.shape[0], 3, replace=False)
      rand_points = cloud[rand_indices]

      # Make a circle with our tree random points
      center, radius = define_circle(rand_points[0], rand_points[1], rand_points[2])

      # TODO: test out if these values are correct to use
      # Try again if center or radius is bad
      if center is None or radius < 0.1 or radius > 0.35:
        continue
 
      # Calculate which points are close to the circle
      circle_offset = 0.05
      center_distances = self.calculate_circle_ditances(cloud, center)
      circle_points = self.calculate_close_points(cloud, center_distances, radius + circle_offset, radius - circle_offset)

      # Save the circle
      saved_ransac_circles = np.append(saved_ransac_circles, Circle(center, radius, circle_points))

    # Delete the mock Cricle
    saved_ransac_circles = np.delete(saved_ransac_circles, 0, axis=0)
    
    # Calculate the variance of the centers
    all_centers = [c.center for c in saved_ransac_circles]
    variance = np.mean(np.var(all_centers, axis=0))

    # Find the circle with the most points
    best_circle_index = np.argmax(saved_ransac_circles)
    best_circle = saved_ransac_circles[best_circle_index]

    # TODO: check the value of the radius, and set better rules afterwards
    print('radius in best circle:', best_circle.radius)

    # Return the best circle and the variace of the centers
    return best_circle, variance
     
def visualize_10_trees(ransac):
  # # # Do RANSAC several times and save the results
  trees = np.zeros((1,3))
  bounding_boxes = []
  tree_centers = []
  i = 0
  while i < 10:
    # Run the calculations
    tree, tree_part = ransac.run_calculations()
  
    # prevents it from happeing
    found_before = False
    for c in range(len(tree_centers)):
      if np.sqrt(np.sum([(tree_part.center[v] - tree_centers[c][v])**2 for v in [0,1]])) < 0.2:
        found_before = True

    if found_before:
      print('Tree found before...')
      continue

    diameter = tree_part.radius * 2
    # Show progress
    print("-"*5, "Found tree", i, "-"*5)
    print('Number of points:', tree.shape[0])
    print('Diameter        :', diameter)

    # Elevate roof points to make them visible
    # tree = tree + [0, 0, 0.01]
    trees = np.append(trees, tree, axis=0)
    
    # Store center to be able to detect duplicates
    tree_centers.append(tree_part.center)

    # Create and store the BB
    o3d_tree = o3d.geometry.PointCloud()
    o3d_tree.points = o3d.utility.Vector3dVector(tree)
    bb = o3d_tree.get_oriented_bounding_box()
    
    # TODO: Get bb dimentions end print them out
    print('Volume          :', bb.volume())
    max_bound = bb.get_max_bound()
    min_bound = bb.get_min_bound()
    print('Height          :', max_bound[2] - min_bound[2])


    bounding_boxes.append(bb)
    i += 1

  print("-"*5, "DONE", "-"*5)
  trees = np.delete(trees, 0, axis=0)

  # Generate o3d cloud
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(trees)

  # cloud.paint_uniform_color([0.1, 0.1, 0.1])

  # TODO: set up a BB around each tree

  # a = cloud.get_oriented_bounding_box
  # a = cloud.get_axis_aligned_bounding_box
  # add a in the list to draw

  # Visualize the cloud
  o3d.visualization.draw_geometries([cloud] + bounding_boxes)

  # Visualize with pptk
  # v = pptk.viewer(trees, trees[:, 2])

  # v.color_map('summer', scale=None)
  # v.color_map('gray')
  # v.set(point_size=0.01)

def visualize_good_slice(ransac):
  # Terrain is already removed by using Cloud compare with the package CSF filter
  # Sort cloud and store for later
  sorted_cloud = ransac.sort_cloud(ransac.xyz)

  # Slice point cloud in horizontal direction
  num_slices = 20
  slices = ransac.slice_cloud(sorted_cloud, num_slices)

  # Generate o3d cloud
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(slices[0])
  
  # Visualize the cloud
  o3d.visualization.draw_geometries([cloud])

def visualize_bad_stem_cut(ransac):
  sorted_cloud = ransac.sort_cloud(ransac.xyz)
  num_slices = 5
  slices = ransac.slice_cloud(sorted_cloud, num_slices)

  cloud = slices[0]

  # Find a random initial point
  indx = np.random.randint(0, cloud.shape[0], 1)
  rand_point = cloud[indx]

  # Find the distnces to all other points in the cloud
  # Only take the distance on x and y, not z
  point_distances = np.sqrt(np.sum((cloud[:,:2] - rand_point[:,:2])**2, axis=1))

  # Find close points to the random point
  max_point_distance = 0.5
  close_points = ransac.calculate_close_points(cloud, point_distances, max_point_distance)

  # Calculate the center of the points
  center = np.mean(close_points[:,:2], axis=0)

  # A tree should be a circle with many points in the circle, but few or none points outside the circle
  center_distances = ransac.calculate_circle_ditances(close_points, center)
  max_circle_distance = np.max(center_distances)

  circle_points = ransac.calculate_close_points(close_points, center_distances, max_circle_distance, max_circle_distance*0.3)

  # Generate o3d cloud
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(circle_points)
  
  # Visualize the cloud
  o3d.visualization.draw_geometries([cloud])

def visualize_cloud(ransac):
  # Generate o3d cloud
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(ransac.xyz)

  # Visualize the cloud
  o3d.visualization.draw_geometries([cloud])

if __name__ == "__main__":
  # Initialice the class
  ransac = RANSAC()

  # visualize_10_trees(ransac)
  visualize_good_slice(ransac)
  # visualize_bad_stem_cut(ransac)
  # visualize_cloud(ransac)

  

