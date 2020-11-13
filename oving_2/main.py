from laspy.file import File
import numpy as np
import open3d as o3d
import pptk


class RANSAC():
  """
  docstring
  """
  def __init__(self):
    # Preprocess
    # Read data from file
    dataFile = File('trees_no_ground.las', mode='r')
    self.max = dataFile.header.max
    self.min = dataFile.header.min
    self.xyz = np.vstack((dataFile.x, dataFile.y, dataFile.z)).transpose()

  def run_calculations(self):
    # Remove terrain
    terrain_removed = self.remove_terrain()

    # Slice point cloud in horizontal direction

    # RANSAC on circles in a slice in 2D by disregarding the z-value

    # Detect the "same" circles in at least 3 layers

    # Set op a 3D bounding box around the tree stems

    # Export the point clouds inside the bounding boxes
    return self.xyz

  def remove_terrain(self):
    return self.xyz

  def slice_cloud(self, cloud):
    slices = np.array(cloud)
    return slices
  
  def RANSAC(self, sliced_cloud):
    return sliced_cloud


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
  v.set(point_size=0.01)




