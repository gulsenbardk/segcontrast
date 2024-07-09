import numpy as np
import torch
import random
import open3d as o3d
import ipdb
import pandas as pd
######################################################################
 
class universe:
    def __init__(self, n_elements):
        self.num = n_elements
        self.elts = np.empty(shape=(n_elements, 3), dtype=int)
        for i in range(n_elements):
            self.elts[i, 0] = 0  # rank
            self.elts[i, 1] = 1  # size
            self.elts[i, 2] = i  # p
 
    def size(self, x):
        return self.elts[x, 1]
 
    def num_sets(self):
        return self.num
 
    def find(self, x):
        y = int(x)
        while y != self.elts[y, 2]:
            y = self.elts[y, 2]
        self.elts[x][2] = y
        return y
 
    def join(self, x, y):
        # x = int(x)
        # y = int(y)
        if self.elts[x, 0] > self.elts[y, 0]:
            self.elts[y, 2] = x
            self.elts[x, 1] += self.elts[y, 1]
        else:
            self.elts[x, 2] = y
            self.elts[y, 1] += self.elts[x, 1]
            if self.elts[x, 0] == self.elts[y, 0]:
                self.elts[y, 0] += 1
        self.num -= 1
 
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb
 
def get_threshold(size, c):
    return c / size
 
def segment_graph(num_vertices, num_edges, edges, c):
    # sort edges by weight (3rd column)
    edges[0:num_edges, :] = edges[edges[0:num_edges, 2].argsort()]
    # make a disjoint-set forest
    u = universe(num_vertices)
    # init thresholds
    threshold = np.zeros(shape=num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)
 
    # for each edge, in non-decreasing weight order...
    for i in range(num_edges):
        pedge = edges[i, :]
        print(pedge[0], pedge[1], pedge[2])
        # components connected by this edge
        a = u.find(pedge[0])
        b = u.find(pedge[1])
        if a != b:
            if (pedge[2] <= threshold[a]) and (pedge[2] <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = pedge[2] + get_threshold(u.size(a), c)
 
    return u
######################################################################
 
 
class Point:
    def __init__(self, coordinates, prediction, features, neighbors):
        self.coordinates = coordinates
        self.prediction = prediction
        # self.hdbscan = hdbscan
        self.features = features
        self.neighbors = neighbors
        self.color = (128, 128, 128)
        self.ins_id = 0 
 
    def set_id(self, point_id):
        self.id = point_id
 
 
def getDistanceMatrix(pcd_coordinates):
   #Distance = np.array([[0 for _ in range(len(pcd_coordinates))] for _ in range(len(pcd_coordinates))])
   Distance = torch.cdist(torch.tensor(pcd_coordinates), torch.tensor(pcd_coordinates)).numpy()
 
   return Distance
 
def getDistanceMatrixbyFeatures(features):
   #Distance = np.array([[0 for _ in range(len(pcd_coordinates))] for _ in range(len(pcd_coordinates))])
   Distance = torch.cdist(torch.tensor(features), torch.tensor(features)).numpy()
   #norm_feats = torch.nn.functional.normalize(torch.from_numpy(features)).numpy()
   #Distance = (norm_feats @ norm_feats.T)
 
   return Distance
 
def CreatePoint(pcd, pcd_predictions, pcd_features, graph, neighbor_num):
    Points = [] 
    for point in range(len(pcd)):
       coordinate = pcd[point]
       prediction = pcd_predictions[point]
    #    hdbscan = pcd_hdbscan[point]
       feature = pcd_features[point]
       idx = np.argsort(graph[point])
       neighbors = idx[1:neighbor_num]
       point_ = Point(coordinate, prediction, feature, neighbors)
       point_.set_id(point)
       Points.append(point_)
 
    # we can do this in the same for loop on the line (point_.set_id(point))
    #for i, point in enumerate(Points):
    #    point.set_id(i)
 
    return Points
 
 
class Edge:
    def __init__(self, nodeTo, nodeFrom, cost):
        self.nodeTo = nodeTo
        self.nodeFrom = nodeFrom
        self.cost = cost
 
    def set_id(self, edge_id):
        self.id = edge_id
 
 
 
def cost(nodeToFeatures, nodeFromFeatures):
    a = nodeToFeatures
    b = nodeFromFeatures
    dot = np.multiply(a,b)
    dot = np.sum(dot, axis=-1)
    a_ = np.sqrt(np.sum(a**2, axis=-1))
    b_ = np.sqrt(np.sum(b**2, axis=-1))
    sim = 1. - dot / (a_ * b_)
    return sim
 
def CreateEdges(Points):
    Edges_ = []
    EdgeNull = []
 
    for point in range(len(Points)):
        neighbors = Points[point].neighbors
        for neighbor in neighbors:
 
            cost_ = cost(Points[point].features, Points[neighbor].features)
            EdgeNull.append([0, 1, point])
            Edge_ = Edge(Points[point], Points[neighbor],cost_)
            Edges_.append(Edge_)
 
 
    EdgeNull = np.asanyarray(EdgeNull)
 
    return Edges_, EdgeNull
 
 
def SegmentGraph(Points, Edges, c):
    #ipdb.set_trace()
 
    num_vertices = len(Points)
    u = universe(num_vertices)
    threshold = np.zeros(shape=num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)
 
    for edge in range(len(Edges)):
       pedge = Edges[edge, :] 
       a = u.find(pedge[0])  
       b = u.find(pedge[1])
       if a != b:
        if (pedge[2] <= threshold[a]) and (pedge[2] <= threshold[b]):
            u.join(a, b)
            a = u.find(a)
            threshold[a] = pedge[2] + get_threshold(u.size(a), c)
    return u
 

 
def Segment(pcd_coordinates, Points, Edges, EdgeNull, c, min_size,class_num):
 
    num_edges = len(Edges)
    num_vertices = len(Points)
 
    if(num_edges != len(EdgeNull)):
        print("error!")
 
    u = SegmentGraph(Points, EdgeNull, c)
 
    edges = np.zeros(shape=(num_edges, 3), dtype=object)
    num = 0
    for edge in range(len(Edges)):
        edges[edge, 0] = Edges[edge].nodeTo.id
        edges[edge, 1] = Edges[edge].nodeFrom.id
        edges[edge, 2] = Edges[edge].cost
        #print(edge, Edges[edge].cost)
 
        a = u.find(edges[edge, 0])
        b = u.find(edges[edge, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)
 
    #u = SegmentGraph(Points, EdgeNull, c)
 
    #for i in range(num):
 
    #    a = u.find(edges[i, 0])
    #    b = u.find(edges[i, 1])
    #    if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
    #        u.join(a, b)
 
    num_cc = u.num_sets()
    output = np.zeros(shape=(num_vertices, 3))
    colors = np.zeros(shape=(num_vertices, 3))
    null_ids = np.zeros(shape=(num_vertices, 1))
    ids = np.zeros(shape=(num_vertices, 1))
 
 
    for i in range(num_vertices):
        colors[i, :] = random_rgb()
        null_ids[i, :] = i
        # null_ids[i, :] = np.random.randint(0,750)
 
    for point in range(num_vertices):
        comp = u.find(point)
        output[point, :] = colors[comp, :]
        ids[point, :] = null_ids[comp, :]
        Points[point].ins_id = ids[point]
        Points[point].color = output[point]
 
    instance_ids = ids.astype(int)
 
    # ipdb.set_trace()
 
 
    return pcd_coordinates, instance_ids , output
 
def GraphBased(points, labels, features, neighbors_num, k, min_size, label):
    # ipdb.set_trace()
    ins_pred_all = np.zeros_like(labels) 
    colors_all = np.zeros_like(points) 
    features = pd.DataFrame(features).to_numpy()
 
 
    labeled_points = points[np.where(labels == label)[0]]
    labeled_preds = labels[np.where(labels == label)[0]]
    labeled_features = features[np.where(labels == label)[0]]
 
 
    if len(labeled_points) > 1: 
        Graph_ = getDistanceMatrix(labeled_points)
        Points = CreatePoint(labeled_points,labeled_preds, labeled_features, Graph_, neighbors_num)
        Edges, EdgeNull = CreateEdges(Points)    
        pcd_coordinates, ins_pred, output_colors = Segment(labeled_points, Points, Edges, EdgeNull, k, min_size, label)
        # clusterer.fit(labeled_points)
        # ins_pred = clusterer.labels_.copy()
        unique_ins_pred = np.unique(ins_pred)
        unique_ins_pred += (label * 10)
 
        ins_pred = ins_pred.reshape(-1)
        ins_pred_all[np.where(labels == label)[0]] = ins_pred + (label * 10)
        colors_all[np.where(labels == label)[0]] = output_colors
 
    else: 
        ins_pred = label
        ins_pred_all[np.where(labels == label)[0]] = ins_pred + (label * 10)
 
    return points, ins_pred_all, colors_all
 
 
class NewSemanticClass():
    def __init__(self, coordinates, features, gt_labels, pegbis_labels, threshold, newsemanticlass_id):
        self.coordinates = coordinates
        self.features = features
        self.gt_labels = gt_labels
        self.pegbis_labels = pegbis_labels
        self.threshold = threshold
        self.color = [128, 128, 128]
        self.ins_id = 0

    def set_id(self, features, gt_labels, pegbis_labels, threshold):
        self.id = newsemanticlass_id
    #newsemanticclass_id.set_id(features, gt_labels, pegbis_labels, threshold)

    
    def set_label(self, similar_label1, similar_label2):

        self.label_with_similarity1 = similar_label1
        self.label_with_similarity2 = similar_label2


def clustering_pegbis_test(points, labels, features):

    # points, labels, pcd_hdbscan, features = getData(path_coordinates, path_features)

    neighbors_num = 16
    k = 300
    min_size = 30 
    
    unique_labels = np.unique(labels)
    ins_pred_all = np.zeros_like(labels) 
    colors_all = np.zeros_like(points) 


    classes = [1,2,3,4,5,6,7,8,14,16,18,19,20]
    # classes = [20]
    for label in classes: #for concatenating known and unknown
    # for label in range(1,9):
        # ipdb.set_trace()
        #print("Class Number PEGBIS: ", label)
        labeled_points = points[np.where(labels == label)[0]]
        labeled_preds = labels[np.where(labels == label)[0]]
        labeled_features = features[np.where(labels == label)[0]]


        if len(labeled_points) > 1: 
           
            Graph_ = getDistanceMatrix(labeled_points)
            Points = CreatePoint(labeled_points,labeled_preds, labeled_features, Graph_, neighbors_num)
            Edges, EdgeNull = CreateEdges(Points)    
            pcd_coordinates, ins_pred, output_colors = Segment(labeled_points, Points, Edges, EdgeNull, k, min_size, label)
          

            unique_ins_pred = np.unique(ins_pred)
            unique_ins_pred += (label * 2)
      
            ins_pred = ins_pred.reshape(-1)
            ins_pred_all[np.where(labels == label)[0]] = ins_pred + (label * 2)
            colors_all[np.where(labels == label)[0]] = output_colors
           

        else: 
            ins_pred = label
            ins_pred_all[np.where(labels == label)[0]] = ins_pred + (label * 2)


    return points, ins_pred_all, colors_all
