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
        # voxelgrid.plot(d=3, mode="density", cmap="hsv")
        print("Total number of points:", len(pcd.points))
        print("Total number of voxels:", self.voxelgrid.n_voxels)
        self.occupied_voxels = np.unique(self.voxelgrid.voxel_n)
        print("Occupied voxels:", len(self.occupied_voxels))

        #Feature calculation - bruke PCA og finne ut hva hver og en voxel er
        self.pca = PCA(n_components=3) #tre dimensjoner
        self.pcd_array = np.asarray(pcd.points)

        self.all_points = o3d.geometry.PointCloud()
        self.all_points.points = o3d.utility.Vector3dVector(self.pcd_array[:,:3])
        # self.all_points.paint_uniform_color([0.1, 0.1, 0.1])
        # colors = [[0, 0, 0.1*j] for j in range(10)]
        # all_points.colors = o3d.utility.Vector3dVector(colors)

        self.num_features = 10
        self.features = np.zeros((self.occupied_voxels.shape[0], self.num_features))
        self.feature_names = ['Linearity', 'Planarity', 'Scattering', 'Omnivariance', '∑ eigenvalues', 'Anisotropy', 'Eigenentropy', '∆ curvature', '2D z-range', 'STD from plane'] 
        self.mean_and_std = np.zeros((self.num_features,3,2))
        
        if self.grid_size == 8:
            # For grid_size 8
            self.selected_feature_indecies = [1, 2, 3, 7]
        else:
            # For grid_size 4
            self.selected_feature_indecies = [1, 2, 3, 5, 6, 7, 9]

        # For all features
        # self.selected_feature_indecies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # TODO: sett riktig...
        self.selected_feature_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

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

        # Print out some stats
        total_points_classified = building_points.shape[0] + terrain_points.shape[0] + tree_points.shape[0]
        building_percentage = building_points.shape[0] / total_points_classified
        terrain_percentage = terrain_points.shape[0] / total_points_classified
        tree_percentage = tree_points.shape[0] / total_points_classified
        print('Number of building points:', building_points.shape[0], '= ' + str(building_percentage) + '%')
        print('Number of terrain points :', terrain_points.shape[0], '= ' + str(terrain_percentage) + '%')
        print('Number of tree points    :', tree_points.shape[0], '= ' + str(tree_percentage) + '%')

        # Visualize the result
        # Will only visualize the classified voxels, not voxels with too few points in it (<10)
        self.visualize_all_voxels_with_class(building_points, terrain_points, tree_points)

    def get_voxel_indecies(self):
        if self.grid_size == 8:
            # for grid_size = 8
            building_voxel_indecies = [114, 115, 121, 999, 1009, 1015, 1029, 1031, 2723, 2724]
            terrain_voxel_indecies = [102, 104, 105, 106, 110, 123, 125, 126, 2729, 2732]
            tree_voxel_indecies = [103, 2701, 2704, 2705, 2708, 2709, 2711, 2712, 2716, 2717]
            return building_voxel_indecies, terrain_voxel_indecies, tree_voxel_indecies

        # for grid_size = 4
        # fine
        building_voxel_indecies = [327, 443, 454, 560, 559, 712, 1405, 2997, 3172, 3174, 3190, 3206, 5089, 8151]
        terrain_voxel_indecies = [213, 300, 301, 401, 449, 525, 530, 1413, 1414, 2991, 3164, 5080, 5104, 5117]
        tree_voxel_indecies = [208, 211, 708, 5002, 5009, 8083, 8088, 8092, 8108, 8113, 8120, 11175, 11176, 11155]
        
        # grove
        # building_voxel_indecies = [46, 63, 129, 398, 499, 500]
        # terrain_voxel_indecies = [1, 2, 11, 13, 121, 495]
        # tree_voxel_indecies = [21, 260, 261, 404, 405]
        # ekstra grove: 11, 63, 398

        return building_voxel_indecies, terrain_voxel_indecies, tree_voxel_indecies

    def select_features(self):
        # Feature selection 
        # Find out which values 
        building_voxel_indecies, terrain_voxel_indecies, tree_voxel_indecies = self.get_voxel_indecies()
        
        for i, voxel in enumerate(self.occupied_voxels): #gå gjennom voxelsene som er bruk
            tmp = np.where(self.voxelgrid.voxel_n == voxel)
            #Liste med indexen til alle punktene som har samme index som den voxelen vi er i nå
            voxel_points = self.pcd_array[tmp]
            #legger til alle punktene i liste

            # Må ha nok punkter i voxelen
            if len(tmp[0]) > 10: 
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
                
                # Liste med alle featursene
                fs = [L, P, S, O, sum_ev, A, E, change_curvature, z_range, plane_std] 
                
                self.features[i] = fs

                # if i in building_voxel_indecies:
                #     self.visualize_voxel_on_cloud(voxel_points, i)

                # if i >= 1800:
                #     self.visualize_voxel_on_cloud(voxel_points, i)
        
        # buildings: [129, 132, 151, 152, 682, 684, 712, 713, 716, 725, 726, 743, 745, ]
        # terrain  : [5, 80, 81, 96, 97, 104, 114, 156, 184, 555, 727, 731, 744, 754, ]
        # trees    : [86, 87, 89, 90, 103, 171, 173, 174, 764]

        # Nomalize plane_std
        max_plane_std = np.max(self.features[:, 9])
        self.features[:, 9] /= max_plane_std

        # Normalize sum_ev
        max_sum_ev = np.max(self.features[:, 4])
        self.features[:, 4] /= max_sum_ev

        # [L, P, S, O, sum_ev, A, E, change_curvature, z_range, plane_std] 
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

            # self.mean_and_std[feature_index, class_index, 0 for mean eller 1 for std]
            
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
        for i, feature_index in enumerate(self.selected_feature_indecies):
            # Give points based on how close (in value, not index) the voxel is 
            # to corespoding mean value for each class
            # Multiply this value by the std to make the values more accurate
            # 1 - this value means it will be higher when:
            # * the voxel-feature-value is closer to the stored mean feature value
            # * the smaller the stored std feature value
            for class_index in range(len(classes)):
                # Skaler dist likt som 0-np.max(self.mean_and_std[feature_index, class_index, 0]) skaleres til 0-1
                # Blir en form for normalisering, sånn at hver featue teller likt, og at de med lave tall ikke teller mye mer
                dist = np.abs(self.features[voxel_index, feature_index] - self.mean_and_std[feature_index, class_index, 0])
                value = dist*self.mean_and_std[feature_index, class_index, 1]
                class_points[class_index] += (1 - value)*self.selected_feature_weights[i]
        
        building_voxel_indecies, terrain_voxel_indecies, tree_voxel_indecies = self.get_voxel_indecies()
        determined_class = classes[np.argmax(class_points)]

        if voxel_index in building_voxel_indecies and determined_class != classes[0]:
            print('Voxel number', voxel_index, 'classified as', determined_class, 'but should be building. Points:', class_points)
        elif voxel_index in terrain_voxel_indecies and determined_class != classes[1]:
            print('Voxel number', voxel_index, 'classified as', determined_class, 'but should be terrain.  Points:', class_points)
        elif voxel_index in tree_voxel_indecies and determined_class != classes[2]:
            print('Voxel number', voxel_index, 'classified as', determined_class, 'but should be tree.     Points:', class_points)

        return determined_class

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

    def select_class_indecies(self):
        ''' Hjelpefunksjon for å finne gode indexer til forskjellige klasser '''
        # self.select_features()

        random_indecies = np.random.choice(self.occupied_voxels.shape[0], 80, replace=False)

        building_voxel_indecies = []
        terrain_voxel_indecies = []
        tree_voxel_indecies = []

        for i, voxel in enumerate(self.occupied_voxels): #gå gjennom voxelsene som er bruk
            if i in random_indecies:
                tmp = np.where(self.voxelgrid.voxel_n == voxel)
                #Liste med indexen til alle punktene som har samme index som den voxelen vi er i nå
                voxel_points = self.pcd_array[tmp]
                #legger til alle punktene i liste

                # Må ha nok punkter i voxelen
                if len(tmp[0]) > 10: 
                    self.visualize_voxel_on_cloud(voxel_points, i)
                    klasse = input('Hvilken klasse er voxelen? [Building: 0, Terrain: 1, Tree: 2, None: 3]:')
                    while not klasse in ['0', '1', '2', '3']:
                        klasse = input('Hvilken klasse er voxelen? [Building: 0, Terrain: 1, Tree: 2, None: 3]:')
                    if klasse != '3':
                        if klasse == '0':
                            building_voxel_indecies.append(i)
                        elif klasse == '1':
                            terrain_voxel_indecies.append(i)
                        elif klasse == '2':
                            tree_voxel_indecies.append(i)
                        print('building indecies:', building_voxel_indecies)
                        print('terrain indecies :', terrain_voxel_indecies)
                        print('tree indecies    :', tree_voxel_indecies)
                        print(len(building_voxel_indecies), len(terrain_voxel_indecies), len(tree_voxel_indecies))     

    def plot_feature_mean_and_std(self):
        # Mostly taken from example:
        # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
        
        def autolabel(rects, ax):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        def make_plot(values, value_name):
            x = np.arange(len(self.selected_feature_indecies))  # the label locations
            width = 0.3  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width, values[0], width, label='Building')
            rects2 = ax.bar(x, values[1], width, label='Terrain')
            rects3 = ax.bar(x + width, values[2], width, label='Tree')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(value_name)
            ax.set_title(f'{value_name} by feature and class with grid size {self.grid_size}')
            ax.set_xticks(x)
            ax.set_xticklabels([self.feature_names[i] for i in self.selected_feature_indecies])
            ax.legend()

            autolabel(rects1, ax)
            autolabel(rects2, ax)
            autolabel(rects3, ax)

            plt.setp(ax.get_xticklabels(), fontsize=10, rotation=-45)

            fig.tight_layout()

            plt.show()

        building_means = [self.mean_and_std[i, 0, 0] for i in self.selected_feature_indecies]
        terrain_means = [self.mean_and_std[i, 1, 0] for i in self.selected_feature_indecies]
        tree_means = [self.mean_and_std[i, 2, 0] for i in self.selected_feature_indecies]

        make_plot([building_means, terrain_means, tree_means], 'Means')
        
        building_std = [self.mean_and_std[i, 0, 1] for i in self.selected_feature_indecies]
        terrain_std = [self.mean_and_std[i, 1, 1] for i in self.selected_feature_indecies]
        tree_std = [self.mean_and_std[i, 2, 1] for i in self.selected_feature_indecies]

        make_plot([building_std, terrain_std, tree_std], 'STD')

    def visualize_cloud(self):
        o3d.visualization.draw_geometries([self.all_points])

    def visualize_voxalization(self):
        self.all_points.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=self.pcd_array[:,:3].shape))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.all_points, voxel_size=self.grid_size)
        o3d.visualization.draw_geometries([voxel_grid])

    def visualize_voxel_on_cloud(self, voxel_points, i):
        print(i)
        vox = o3d.geometry.PointCloud()
        vox.points = o3d.utility.Vector3dVector(voxel_points[:,:3] + [0,0,0.01])
        vox.paint_uniform_color([0.1, 0.1, 0.1])
        # colors = [[0, 0, 0.1*j] for j in range(10)]
        # vox.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([self.all_points, vox])

    def visualize_all_voxels_with_class(self, building_points, terrain_points, tree_points):
        # First as points
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

        # Then as voxesl
        voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(tree, voxel_size=self.grid_size)
        voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(terrain, voxel_size=self.grid_size)
        voxel_grid3 = o3d.geometry.VoxelGrid.create_from_point_cloud(building, voxel_size=self.grid_size)
        o3d.visualization.draw_geometries([voxel_grid1, voxel_grid2, voxel_grid3])

    def visualize_voxel_indecies(self, building_voxel_indecies, terrain_voxel_indecies, tree_voxel_indecies):

        # Initialize the point arrays
        # building_points = [self.pcd_array[np.where(self.voxelgrid.voxel_n == self.occupied_voxels[i])][:,:3] for i in building_voxel_indecies]
        # terrain_points = [self.pcd_array[np.where(self.voxelgrid.voxel_n == self.occupied_voxels[i])][:,:3] for i in terrain_voxel_indecies]
        # tree_points = [self.pcd_array[np.where(self.voxelgrid.voxel_n == self.occupied_voxels[i])][:,:3] for i in tree_voxel_indecies]
        
        # Initialize the point arrays
        building_points = np.zeros((1,3))
        terrain_points = np.zeros((1,3))
        tree_points = np.zeros((1,3))
        
        nums = [0,0,0]

        for i, voxel in enumerate(self.occupied_voxels):
            if i in building_voxel_indecies:
                tmp = np.where(self.voxelgrid.voxel_n == voxel)
                voxel_points = self.pcd_array[tmp]
                # print('new building voxel:', voxel_points.shape[0])
                building_points = np.append(building_points, voxel_points[:,:3], axis=0)
                nums[0] += 1
            elif i in terrain_voxel_indecies:
                tmp = np.where(self.voxelgrid.voxel_n == voxel)
                voxel_points = self.pcd_array[tmp]
                # print('new terrain voxel:', voxel_points.shape[0])
                terrain_points = np.append(terrain_points, voxel_points[:,:3], axis=0)
                nums[1] += 1
            elif i in tree_voxel_indecies:
                tmp = np.where(self.voxelgrid.voxel_n == voxel)
                voxel_points = self.pcd_array[tmp]
                # print('new tree voxel:', voxel_points.shape[0])
                tree_points = np.append(tree_points, voxel_points[:,:3], axis=0)
                nums[2] += 1
        # Delete the initial zeros
        building_points = np.delete(building_points, 0, axis=0)
        terrain_points = np.delete(terrain_points, 0, axis=0)
        tree_points = np.delete(tree_points, 0, axis=0)

        print('Number of voxels:', nums)

        building = o3d.geometry.PointCloud()
        building.points = o3d.utility.Vector3dVector(building_points + [0,0,0.1])
        building.paint_uniform_color([1, 0, 0])

        terrain = o3d.geometry.PointCloud()
        terrain.points = o3d.utility.Vector3dVector(terrain_points + [0,0,0.1])
        terrain.paint_uniform_color([0, 1, 0])

        tree = o3d.geometry.PointCloud()
        tree.points = o3d.utility.Vector3dVector(tree_points + [0,0,0.1])
        tree.paint_uniform_color([0, 0, 1])

        self.all_points.paint_uniform_color([0.1, 0.1, 0.1])

        # Visualize with and without backgound points
        o3d.visualization.draw_geometries([self.all_points, building, terrain, tree])
        o3d.visualization.draw_geometries([building, terrain, tree])

    def evaluate_voxel_classes(self):
        building_indecies = [129, 132, 151, 152, 682, 684, 712, 713, 716, 725, 726, 743, 745, 1787, 1789, 1805, 1807, 1810, 1815, 4029]
        terrain_indecies  = [32, 1967, 3296, 3352, 3366, 3477, 4098, 4183, 4187, 5050, 5360, 5794, 6136, 6389, 7228, 7623, 8094, 9463, 10217, 11066, 11300, 11333, 12032, 12077, 12129, 12546, 12812]
        tree_indecies     = [86, 173, 764, 5257, 9572, 9653, 5363, 5634, 7735, 8444, 8457, 5755, 6573, 6668, 7328, 8496, 8527, 8631, 8990, 9399, 9790, 10069, 10244, 10389, 10490, 10567, 10758, 11059, 11131, 11159, 11176, 11697, 12907, 13005, 13453]

        self.visualize_voxel_indecies(building_indecies, terrain_indecies[:20], tree_indecies[:20])        
        
        building_results = []
        terrain_results  = []
        tree_results     = []

        detected_building_indecies = []
        detected_terrain_indecies  = []
        detected_tree_indecies     = []

        for i in range(20):
            guess = self.determine_voxel_class(building_indecies[i])
            building_results.append(guess)
            if guess == 'building':
                detected_building_indecies.append(building_indecies[i])
            elif guess == 'terrain':
                detected_terrain_indecies.append(building_indecies[i])
            elif guess == 'tree':
                detected_tree_indecies.append(building_indecies[i])

            guess = self.determine_voxel_class(terrain_indecies[i])
            terrain_results.append(guess)
            if guess == 'building':
                detected_building_indecies.append(terrain_indecies[i])
            elif guess == 'terrain':
                detected_terrain_indecies.append(terrain_indecies[i])
            elif guess == 'tree':
                detected_tree_indecies.append(terrain_indecies[i])
            
            guess = self.determine_voxel_class(tree_indecies[i])
            tree_results.append(guess)
            if guess == 'building':
                detected_building_indecies.append(tree_indecies[i])
            elif guess == 'terrain':
                detected_terrain_indecies.append(tree_indecies[i])
            elif guess == 'tree':
                detected_tree_indecies.append(tree_indecies[i])

        
        print(building_results.count('building'), terrain_results.count('building'), tree_results.count('building'))
        print(building_results.count('terrain'), terrain_results.count('terrain'), tree_results.count('terrain'))
        print(building_results.count('tree'), terrain_results.count('tree'), tree_results.count('tree'))

        self.visualize_voxel_indecies(detected_building_indecies, detected_terrain_indecies, detected_tree_indecies)


if __name__ == "__main__":
    classifier = Classification()
    classifier.select_features()
    # classifier.plot_feature_mean_and_std()
    # building_voxel_indecies, terrain_voxel_indecies, tree_voxel_indecies = classifier.get_voxel_indecies()
    # classifier.visualize_voxel_indecies(building_voxel_indecies, terrain_voxel_indecies, tree_voxel_indecies)
    # classifier.visualize_cloud()
    # classifier.visualize_voxalization()
    # classifier.select_class_indecies()
    classifier.evaluate_voxel_classes()
    # classifier.run_calculations()
