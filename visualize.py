import open3d as o3d
import numpy as np
import ipdb
import matplotlib.pyplot as plt
import tkinter as tk 
 
def visualize_point_cloud(point_cloud, labels, cmap_dict):
 
    pcd = o3d.geometry.PointCloud()

 
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
 
 
    # ipdb.set_trace()
    colors = np.zeros((len(labels), 3))
 
    for label, color in cmap_dict.items():
 
        indices = np.where(labels == label)[0]
        print("Label", label, "has", len(indices), "points")
        if len(indices) > 0:
            assert np.max(indices) < len(labels), "Label indices out of range"
            colors[indices] = np.array(color)
            print(colors.shape)
 
    colors = colors / 255.0
    bgr_colors = colors[:, ::-1]
    print(bgr_colors.shape)
 
    pcd.colors = o3d.utility.Vector3dVector(bgr_colors)
    o3d.visualization.draw_geometries([pcd])
 
 
# def visualize_instance_labels(point_set, labels):
#     pcd = o3d.geometry.PointCloud()
#     pcd_scans = point_set[0][:, :3]
#     pcd.points = o3d.utility.Vector3dVector(pcd_scans)
 
#     import matplotlib.pyplot as plt
 
#     labels_1d = labels.reshape(-1)
#     colors = plt.get_cmap("prism")(
#         labels / (labels_1d.max() if labels_1d.max() > 0 else 1))
#     colors[labels_1d < 0] = 0
#     ipdb.set_trace()
#     colors_ = colors[:, :, :3]
#     # new_arr = arr[:, :, :3]
# # runtime errors bc of the 3D Array
#     pcd.colors = o3d.utility.Vector3dVector(colors_)
#     o3d.visualization.draw_geometries([pcd])
 
def TARL_visualize_pcd_clusters_gt(x_coord, labels, target_label):
    # Only select points that have the specified target_label
    selected_points = x_coord[labels == target_label]
    pcd = o3d.geometry.PointCloud()
    # pcd_scans = points[0][:, :3]
    # pcd.points = o3d.utility.Vector3dVector(pcd_scans)
 
 
    pcd.points = o3d.utility.Vector3dVector(selected_points)
# flat_indices = np.unique(labels[:,-1]) - original
    colors = np.zeros((len(labels), 4))
    flat_indices = np.unique(labels[:,-1])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("prism")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
        # ipdb.set_trace()
        colors[labels[:,-1] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
    #same here
    colors[labels[:,-1] == -1] = [0.7, 0.7, 0.7, 0.1]
    #and here
    colors_ = colors[:, :3]
 
    pcd.colors = o3d.utility.Vector3dVector(colors_)
 
    o3d.visualization.draw_geometries([pcd])
    
    # Visualization code for selected_points
    # ... (insert your visualization code here)
    # For example, you can use libraries like matplotlib or other visualization tools
    
    # Return the selected points if needed
    # return selected_points




def TARL_visualize_pcd_clusters_instances(points, labels):
    pcd = o3d.geometry.PointCloud()
    # pcd_scans = points[0][:, :3]
    # pcd.points = o3d.utility.Vector3dVector(pcd_scans)
 
 
    pcd.points = o3d.utility.Vector3dVector(points)
# flat_indices = np.unique(labels[:,-1]) - original
    colors = np.zeros((len(labels), 4))
    flat_indices = np.unique(labels[:,-1])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("prism")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
        # ipdb.set_trace()
        colors[labels[:,-1] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
    #same here
    colors[labels[:,-1] == 0] = [0.7, 0.7, 0.7, 0.1]
    #and here
    colors_ = colors[:, :3]
 
    pcd.colors = o3d.utility.Vector3dVector(colors_)
 
    o3d.visualization.draw_geometries([pcd])
 
def TARL_visualize_pcd_clusters(points, labels):
    pcd = o3d.geometry.PointCloud()
    # pcd_scans = points[0][:, :3]
    # pcd.points = o3d.utility.Vector3dVector(pcd_scans)
    pcd.points = o3d.utility.Vector3dVector(points)
    # flat_indices = np.unique(labels[:,-1]) - original
    colors = np.zeros((len(labels), 4))
    flat_indices = np.unique(labels[:])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
        # ipdb.set_trace()
        colors[labels[:] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]

    colors[labels[:] == 0] = [0.7, 0.7, 0.7, 0.1]

    #and here
    colors_ = colors[:, :3]
 
    pcd.colors = o3d.utility.Vector3dVector(colors_)
 
    o3d.visualization.draw_geometries([pcd])

def TARL_visualize_pcd_instances_onelabel(points, labels, target_label):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Filter points and labels based on the target_label
    mask = labels == target_label
    target_points = points[mask]
    target_labels = labels[mask]

    colors = np.zeros((len(target_labels), 4))

    flat_indices = np.unique(target_labels)
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

    for idx in range(len(flat_indices)):
        colors[target_labels == flat_indices[idx]] = colors_instance[idx]

    # Set a default color for points not belonging to the target label
    colors[target_labels != target_label] = [0.7, 0.7, 0.7, 1]

    # Extract RGB color values
    colors_ = colors[:, :3]

    pcd.colors = o3d.utility.Vector3dVector(colors_)

    o3d.visualization.draw_geometries([pcd])


def compare_pcd_clusters3(points1, points2, points3, label1, label2, label3, label1_name, label2_name, label3_name, label_to_visualize):
    root = tk.Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()

    # Filter points with the specific label for dataset 1
    filtered_points = points1[label1 == label_to_visualize]
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(filtered_points)
    pcd1.colors = o3d.utility.Vector3dVector(np.full_like(filtered_points, [1, 0, 0]))

    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window(window_name=f'Dataset 1 - Label {label_to_visualize}', width=int(w/3), height=h, left=0, top=0)
    vis1.add_geometry(pcd1)

    # Filter points with the specific label for dataset 2
    #filtered_points2 = points2[label2 == label_to_visualize]
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(filtered_points)
    pcd2.colors = o3d.utility.Vector3dVector(np.full_like(filtered_points, [0, 1, 0]))

    vis2 = o3d.visualization.VisualizerWithEditing()
    vis2.create_window(window_name=f'Dataset 2 - Label {label_to_visualize}', width=int(w/3), height=h, left=int(w/3), top=0)
    vis2.add_geometry(pcd2)

    # Filter points with the specific label for dataset 3
    #filtered_points3 = points3[label3 == label_to_visualize]
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(filtered_points)
    pcd3.colors = o3d.utility.Vector3dVector(np.full_like(filtered_points, [0, 0, 1]))

    vis3 = o3d.visualization.VisualizerWithEditing()
    vis3.create_window(window_name=f'Dataset 3 - Label {label_to_visualize}', width=int(w/3), height=h, left=int(2*w/3), top=0)
    vis3.add_geometry(pcd3)

    while True:
        vis1.update_geometry(pcd1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(pcd2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(pcd3)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

        sync_view_controls3(vis1, vis2, vis3)
    
    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()


def TARL_visualize_compare_pcd_clusters3(points1, points2, points3, label1, label2, label3, label1_name, label2_name, label3_name):
 
 
    root = tk.Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    
    #ipdb.set_trace()
 
 
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
 
    colors = np.zeros((len(label1), 4))
    flat_indices = np.unique(label1[:])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
 
        colors[label1[:] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
 
    colors[label1[:] == 0] = [0.7, 0.7, 0.7, 0.1]
 
    colors_ = colors[:, :3]
 
    pcd1.colors = o3d.utility.Vector3dVector(colors_)
 
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window(window_name=label1_name, width=int(w/3), height=h, left=0, top=0)
    vis1.add_geometry(pcd1)
 
 
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
 
    colors = np.zeros((len(label2), 4))
    flat_indices = np.unique(label2[:])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
 
        colors[label2[:] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
 
    colors[label2[:] == 0] = [0.7, 0.7, 0.7, 0.1]
 
    colors_ = colors[:, :3]
 
    pcd2.colors = o3d.utility.Vector3dVector(colors_)
 
    vis2 = o3d.visualization.VisualizerWithEditing()
    vis2.create_window(window_name=label2_name, width=int(w/3), height=h, left=int(w/3), top=0)
    vis2.add_geometry(pcd2)
 
    #####################################################################################
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(points3)
 
    colors = np.zeros((len(label3), 4))
    flat_indices = np.unique(label3[:])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
 
        colors[label3[:] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
 
def TARL_visualize_compare_pcd_clusters2(points1, points2, label1, label2, label1_name, label2_name):
 
 
    root = tk.Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    
    # ipdb.set_trace()
 
 
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
 
    colors = np.zeros((len(label1), 4))
    flat_indices = np.unique(label1[:])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
 
        colors[label1[:] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
 
    colors[label1[:] == 0] = [0.7, 0.7, 0.7, 0.1]
 
    colors_ = colors[:, :3]
 
    pcd1.colors = o3d.utility.Vector3dVector(colors_)
 
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window(window_name=label1_name, width=int(w/2), height=h, left=0, top=0)
    vis1.add_geometry(pcd1)
 
 
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
 
    colors = np.zeros((len(label2), 4))
    flat_indices = np.unique(label2[:])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
 
        colors[label2[:] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
 
    colors[label2[:] == 0] = [0.7, 0.7, 0.7, 0.1]
 
    colors_ = colors[:, :3]
 
    pcd2.colors = o3d.utility.Vector3dVector(colors_)
 
    vis2 = o3d.visualization.VisualizerWithEditing()
    vis2.create_window(window_name=label2_name, width=int(w/2), height=h, left=int(w/2), top=0)
    vis2.add_geometry(pcd2)
 
  
 
    while True:
        vis1.update_geometry(pcd1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
 
        vis2.update_geometry(pcd2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
 
       
        sync_view_controls2(vis1, vis2)
    vis1.destroy_window()
    vis2.destroy_window()

 
def sync_view_controls2(vis1, vis2):
    view_control1 = vis1.get_view_control()
    view_control2 = vis2.get_view_control()
    view_control1.convert_from_pinhole_camera_parameters(view_control2.convert_to_pinhole_camera_parameters())
    colors[label3[:] == 0] = [0.7, 0.7, 0.7, 0.1]
 
    colors_ = colors[:, :3]
 
    pcd3.colors = o3d.utility.Vector3dVector(colors_)
 
    vis3 = o3d.visualization.VisualizerWithEditing()
    vis3.create_window(window_name=label3_name, width=int(w/3), height=h, left=int(2*w/3), top=0)
    vis3.add_geometry(pcd3)
 
    while True:
        vis1.update_geometry(pcd1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
 
        vis2.update_geometry(pcd2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
 
        vis3.update_geometry(pcd3)
        if not vis3.poll_events():
            break
        vis3.update_renderer()
        sync_view_controls3(vis1, vis2, vis3)
    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()


def sync_view_controls3(vis1, vis2, vis3):
    view_control1 = vis1.get_view_control()
    view_control2 = vis2.get_view_control()
    view_control3 = vis3.get_view_control()
    view_control1.convert_from_pinhole_camera_parameters(view_control2.convert_to_pinhole_camera_parameters())
    view_control1.convert_from_pinhole_camera_parameters(view_control3.convert_to_pinhole_camera_parameters())
    view_control2.convert_from_pinhole_camera_parameters(view_control1.convert_to_pinhole_camera_parameters())
    view_control2.convert_from_pinhole_camera_parameters(view_control3.convert_to_pinhole_camera_parameters())
    view_control3.convert_from_pinhole_camera_parameters(view_control1.convert_to_pinhole_camera_parameters())
    view_control3.convert_from_pinhole_camera_parameters(view_control2.convert_to_pinhole_camera_parameters())



def TARL_visualize_compare_pcd_clusters2(points1, points2, label1, label2, label1_name, label2_name):
 
 
    root = tk.Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    
    # ipdb.set_trace()
 
 
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
 
    colors = np.zeros((len(label1), 4))
    flat_indices = np.unique(label1[:])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
 
        colors[label1[:] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
 
    colors[label1[:] == 0] = [0.7, 0.7, 0.7, 0.1]
 
    colors_ = colors[:, :3]
 
    pcd1.colors = o3d.utility.Vector3dVector(colors_)
 
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window(window_name=label1_name, width=int(w/2), height=h, left=0, top=0)
    vis1.add_geometry(pcd1)
 
 
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
 
    colors = np.zeros((len(label2), 4))
    flat_indices = np.unique(label2[:])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("tab20b")(
        np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))
 
    for idx in range(len(flat_indices)):
 
        colors[label2[:] ==
               flat_indices[int(idx)]] = colors_instance[int(idx)]
 
    colors[label2[:] == 0] = [0.7, 0.7, 0.7, 0.1]
 
    colors_ = colors[:, :3]
 
    pcd2.colors = o3d.utility.Vector3dVector(colors_)
 
    vis2 = o3d.visualization.VisualizerWithEditing()
    vis2.create_window(window_name=label2_name, width=int(w/2), height=h, left=int(w/2), top=0)
    vis2.add_geometry(pcd2)
 
  
 
    while True:
        vis1.update_geometry(pcd1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
 
        vis2.update_geometry(pcd2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
 
       
        sync_view_controls2(vis1, vis2)
    vis1.destroy_window()
    vis2.destroy_window()


def compare_pcd_clusters2(points1, points2, label1, label2, label1_name, label2_name, label_to_visualize):
    root = tk.Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    
    # Filter points with the specified label for both datasets
    filtered_points1 = points1[label1 == label_to_visualize]
    filtered_points2 = points2[label2 == label_to_visualize]

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(filtered_points1)
    pcd1.colors = o3d.utility.Vector3dVector(np.full_like(filtered_points1, [1, 0, 0]))

    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window(window_name=f'Dataset 1 - Label {label_to_visualize}', width=int(w/2), height=h, left=0, top=0)
    vis1.add_geometry(pcd1)
 
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(filtered_points2)
    pcd2.colors = o3d.utility.Vector3dVector(np.full_like(filtered_points2, [0, 0, 1]))

    vis2 = o3d.visualization.VisualizerWithEditing()
    vis2.create_window(window_name=f'Dataset 2 - Label {label_to_visualize}', width=int(w/2), height=h, left=int(w/2), top=0)
    vis2.add_geometry(pcd2)
 
    while True:
        vis1.update_geometry(pcd1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
 
        vis2.update_geometry(pcd2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
 
        sync_view_controls2(vis1, vis2)
    
    # vis1.destroy_window()
    # vis2.destroy_window()

def sync_view_controls2(vis1, vis2):
    view_control1 = vis1.get_view_control()
    view_control2 = vis2.get_view_control()
    view_control1.convert_from_pinhole_camera_parameters(view_control2.convert_to_pinhole_camera_parameters())