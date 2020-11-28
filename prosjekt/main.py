from pyntcloud import PyntCloud
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#eigenvalues, lengden utstrukket i hver retning
#eigenvector, retning
#Scattering to find treees





class Classification():
    def __init__(self):
        #Voxelization
        pcd = PyntCloud.from_file("cropped.ply")
        self.grid_size = 4
        voxelgrid_id = pcd.add_structure("voxelgrid", size_x=self.grid_size, size_y=self.grid_size, size_z=self.grid_size)
        self.voxelgrid = pcd.structures[voxelgrid_id]
        #voxelgrid.plot(d=3, mode="density", cmap="hsv")
        print("Total number of points:", len(pcd.points))
        print("Total number of voxels:", self.voxelgrid.n_voxels)
        self.occupied_voxels = np.unique(self.voxelgrid.voxel_n)
        print("Occupied voxels:", len(self.occupied_voxels))

        #Feature calculation - bruke PCA og finne ut hva hver og en voxel er
        self.pca = PCA(n_components=3) #tre dimensjoner
        self.pcd_array = np.asarray(pcd.points)

        self.all_points = o3d.geometry.PointCloud()
        self.all_points.points = o3d.utility.Vector3dVector(self.pcd_array[:,:3])
        # all_points.paint_uniform_color([0.1, 0.1, 0.1])
        # colors = [[0, 0, 0.1*j] for j in range(10)]
        # all_points.colors = o3d.utility.Vector3dVector(colors)

        self.num_features = 10
        self.features = np.zeros((self.occupied_voxels.shape[0], self.num_features))
        self.feature_names = ['Linearity', 'Planarity', 'Scattering', 'Omnivariance', '∑ eigenvalues', 'Anisotropy', 'Eigenentropy', '∆ curvature', '2D z-range', 'STD from plane'] 
        self.mean_and_std = np.zeros((self.num_features,3,2))
        self.selected_feature_indecies = [1, 2, 3, 7]

    def run_calculations(self):
        # Find mean and std values for the features
        self.select_features()

        # Plot the values
        # self.plot_feature_mean_and_std()

        #  ---  Classify the voxels  ----

        # Initialize the point arrays
        building_points = np.zeros((1,3))
        terrain_points = np.zeros((1,3))
        tree_points = np.zeros((1,3))

        # Go through each voxel, determine its class and add its 
        # points to the coresponding array of points
        for i, voxel in enumerate(self.occupied_voxels):
            tmp = np.where(self.voxelgrid.voxel_n == voxel)
            #Liste med indexen til alle punktene som har samme index som den voxelen vi er i nå
            voxel_points = self.pcd_array[tmp]
            #legger til alle punktene i liste

            voxel_class = self.determine_voxel_class(i)

            if voxel_class == "building":
                building_points = np.append(building_points, voxel_points[:,:3], axis=0)
            elif voxel_class == "terrain":
                terrain_points = np.append(terrain_points, voxel_points[:,:3], axis=0)
            elif voxel_class == "tree":
                tree_points = np.append(tree_points, voxel_points[:,:3], axis=0)

        # Delete the initial zeros
        building_points = np.delete(building_points, 0, axis=0)
        terrain_points = np.delete(terrain_points, 0, axis=0)
        tree_points = np.delete(tree_points, 0, axis=0)

        # Visualize the result
        # Will only visualize the classified voxels, not voxels with too few points in it (<10)
        self.visualize_all_voxels_with_class(building_points, terrain_points, tree_points)

    def select_features(self):
        # Feature selection 
        # Find out which values 

        # for grid_size = 8
        building_voxel_indecies = [114, 115, 121, 999, 1009, 1015, 1029, 1031, 2723, 2724]
        terrain_voxel_indecies = [102, 104, 105, 106, 110, 123, 125, 126, 2729, 2732]
        tree_voxel_indecies = [103, 2701, 2704, 2705, 2708, 2709, 2711, 2712, 2716, 2717]

        # for grid_size = 4
        # building_voxel_indecies = [327, 444, 445, 446]
        # terrain_voxel_indecies = [213, 300, 301, 449]
        # tree_voxel_indecies = [208, 211, ]
        
        for i, voxel in enumerate(self.occupied_voxels): #gå gjennom voxelsene som er bruk
            tmp = np.where(self.voxelgrid.voxel_n == voxel)
            #Liste med indexen til alle punktene som har samme index som den voxelen vi er i nå
            voxel_points = self.pcd_array[tmp]
            #legger til alle punktene i liste

            if len(tmp[0]) > 10: #må ha punkter som er flere
                self.pca.fit(voxel_points)
                ev = self.pca.explained_variance_
                ev_norm = self.pca.explained_variance_ratio_
                #eigenvalues, lengeden (ratio for å normalisere)

                l1, l2, l3 = ev_norm
                # lagrer lambda1,2 og 3

                L = (l1 - l2) / l1  # Linearity
                P = (l2 - l3) / l1  # Planarity
                S = l3 / l1  # Scattering, sphericity
                O = (l1 * l2 * l3) ** (1 / 3)  # Omnivariance
                sum_ev = ev[0] + ev[1] + ev[2]
                sum_ev_norm = l1 + l2 + l3  # Sum of un-normalized eigenvalues
                A = (l1 - l3) / l1  # Anisotropy
                E = -1 * ((l1 * np.log(l1)) + (l2 * np.log(l2)) + (l3 * np.log(l3)))  # Eigenentropy
                change_curvature = l3 / sum_ev_norm  # Change of curvature

                # De nye featurene
                z_range = (np.max(voxel_points[:,2]) - np.min(voxel_points[:,2])) / self.grid_size

                # Standard deviation from a plane
                plane_std = self.standard_deviation_from_plane(voxel_points, i)
                
                '''
                if i in building_voxel_indecies:
                    self.visualize_voxel_on_cloud(voxel_points, i)
                '''

                # if i >= 430:
                #     self.visualize_voxel_on_cloud(voxel_points, i)

                fs = [L, P, S, O, sum_ev_norm, A, E, change_curvature, z_range, plane_std] #liste med alle
                
                self.features[i] = fs
        
        # [L, P, S, O, sum_ev, A, E, change_curvature] 
        building_mean_features = np.mean([self.features[v_i] for v_i in building_voxel_indecies], axis=0)
        terrain_mean_features = np.mean([self.features[v_i] for v_i in terrain_voxel_indecies], axis=0)
        tree_mean_features = np.mean([self.features[v_i] for v_i in tree_voxel_indecies], axis=0)

        building_std_features = np.std([self.features[v_i] for v_i in building_voxel_indecies], axis=0)
        terrain_std_features = np.std([self.features[v_i] for v_i in terrain_voxel_indecies], axis=0)
        tree_std_features = np.std([self.features[v_i] for v_i in tree_voxel_indecies], axis=0)

        # TODO: Finn ut hvilke som er relevante eller ikke
        # Kriterier, én eller to skiller seg ut og jo lavere std jo bedre! 

        for i in range(self.num_features):
            self.mean_and_std[i, 0] = [building_mean_features[i], building_std_features[i]]
            self.mean_and_std[i, 1] = [terrain_mean_features[i], terrain_std_features[i]]
            self.mean_and_std[i, 2] = [tree_mean_features[i], tree_std_features[i]]
            
            # print(self.feature_names[i] + ':')
            # print('building: mean =', building_mean_features[i], ',\tstd =', building_std_features[i])
            # print('terrain : mean =', terrain_mean_features[i], ',\tstd =', terrain_std_features[i])
            # print('tree    : mean =', tree_mean_features[i], ',\tstd =', tree_std_features[i])
            # print()
    
    def determine_voxel_class(self, voxel_index):
        # Classification
        # TODO: find a good measure for each of the features.
        # Must be fair for each of the features...
        # Go through each selected_feature_indecies

        # class_points = [building_value, terrain_value, tree_value]
        classes = ["building", "terrain", "tree"]
        class_points = [0,0,0]
        for feature_index in self.selected_feature_indecies:
            # Give points based on how close (in value, not index) the voxel is 
            # to corespoding mean value for each class
            # Multiply this value by the std to make the values more accurate
            # 1 - this value means it will be higher when:
            # * the voxel-feature-value is closer to the stored mean feature value
            # * the smaller the stored std feature value
            for i in range(3):
                value = np.abs(self.features[voxel_index, feature_index] - self.mean_and_std[feature_index, i, 0])

                class_points[i] += 1 - value*self.mean_and_std[feature_index, i, 1]

        return classes[np.argmax(class_points)]

    def standard_deviation_from_plane(self, points, i):
        # Plane calculation mostly taken from: 
        # https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6

        # The centroid
        centroid = np.sum(points, axis=0)/points.shape[0]

        # Centered data
        c_data = points - centroid
        A = np.c_[c_data[:,0], c_data[:,1], np.ones(c_data.shape[0])]

        # Plane coefficients
        C,_,_,_ = np.linalg.lstsq(A, c_data[:,2], rcond=None)
        #       Z = C[0]*X + C[1]*Y + C[2]
        # =>    C[0]*X + C[1]*Y -1*Z + C[2] = 0

        # Distances from the plane
        distances = np.abs(C[0]*c_data[:,0] + C[1]*c_data[:,1] -1*c_data[:,2] + C[2]) / np.sqrt(C[0]**2 + C[1]**2 + 1)

        # Normalize distances
        distances /= self.grid_size

        # Return the standard deviation
        return np.std(distances)

    def plot_feature_mean_and_std(self):
        # Mostly taken from example:
        # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
        building_means = [self.mean_and_std[i, 0, 0] for i in self.selected_feature_indecies]
        terrain_means = [self.mean_and_std[i, 1, 0] for i in self.selected_feature_indecies]
        tree_means = [self.mean_and_std[i, 2, 0] for i in self.selected_feature_indecies]

        x = np.arange(len(self.selected_feature_indecies))  # the label locations
        width = 0.3  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, building_means, width, label='Building')
        rects2 = ax.bar(x, terrain_means, width, label='Terrain')
        rects3 = ax.bar(x + width, tree_means, width, label='Tree')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Mean')
        ax.set_title('Mean by feature and class')
        ax.set_xticks(x)
        ax.set_xticklabels([self.feature_names[i] for i in self.selected_feature_indecies])
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        plt.setp(ax.get_xticklabels(), fontsize=10, rotation=-45)

        fig.tight_layout()

        plt.show()

    def visualize_voxel_on_cloud(self, voxel_points, i):
        print(i)
        vox = o3d.geometry.PointCloud()
        vox.points = o3d.utility.Vector3dVector(voxel_points[:,:3] + [0,0,0.01])
        vox.paint_uniform_color([0.1, 0.1, 0.1])
        # colors = [[0, 0, 0.1*j] for j in range(10)]
        # vox.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([self.all_points, vox])

    def visualize_all_voxels_with_class(self, building_points, terrain_points, tree_points):
        building = o3d.geometry.PointCloud()
        building.points = o3d.utility.Vector3dVector(building_points)
        building.paint_uniform_color([1, 0, 0])

        terrain = o3d.geometry.PointCloud()
        terrain.points = o3d.utility.Vector3dVector(terrain_points)
        terrain.paint_uniform_color([0, 1, 0])

        tree = o3d.geometry.PointCloud()
        tree.points = o3d.utility.Vector3dVector(tree_points)
        tree.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([building, terrain, tree])

if __name__ == "__main__":
    classifier = Classification()
    classifier.select_features()
    # classifier.plot_feature_mean_and_std()
    # classifier.run_calculations()
