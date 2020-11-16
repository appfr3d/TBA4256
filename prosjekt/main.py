from pyntcloud import PyntCloud
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA

#eigenvalues, lengden utstrukket i hver retning
#eigenvector, retning
#Scattering to find treees

#Voxelization
pcd = PyntCloud.from_file("finalproject.ply")
voxelgrid_id = pcd.add_structure("voxelgrid", n_x=64, n_y=64, n_z=64)
voxelgrid = pcd.structures[voxelgrid_id]
#voxelgrid.plot(d=3, mode="density", cmap="hsv")
print("Total number of points:", len(pcd.points))
print("Total number of voxels:", voxelgrid.n_voxels)
occupied_voxels = np.unique(voxelgrid.voxel_n)
print("Occupied voxels:", len(occupied_voxels))

#Feature selection - bruke PCA og finne ut hva hver og en voxel er
pca = PCA(n_components=3) #tre dimensjoner
pcd_array = np.asarray(pcd.points)


for voxel in occupied_voxels: #gå gjennom voxelsene som er bruk
   tmp = np.where(voxelgrid.voxel_n == voxel)
   #Liste med indexen til alle punktene som har samme index som den voxelen vi er i nå
   voxel_points = pcd_array[tmp]
   #legger til alle punktene i liste

   if len(tmp[0]) > 2: #må ha punkter som er flere
       pca.fit(voxel_points)
       ev = pca.explained_variance_ratio_
       #eigenvalues, lengeden (ratio for å normalisere)

       l1, l2, l3 = ev
       #lagrer lambda1,2 og 3

       L = (l1 - l2) / l1  # Linearity
       P = (l2 - l3) / l1  # Planarity
       S = l3 / l1  # Scattering, sphericity
       O = (l1 * l2 * l3) ** (1 / 3)  # Omnivariance
       sum_ev = l1 + l2 + l3  # Sum of eigenvalues
       A = (l1 - l3) / l1  # Anisotropy
       E = -1 * ((l1 * np.log(l1)) + (l2 * np.log(l2)) + (l3 * np.log(l3)))  # Eigenentropy
       change_curvature = l3 / sum_ev  # Change of curvature
       fs = [L, P, S, O, sum_ev, A, E, change_curvature] #liste med alle


#Feature calculation

#Classification
