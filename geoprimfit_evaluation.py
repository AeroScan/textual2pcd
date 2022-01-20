import numpy as np
from genericpath import exists
from os.path import join, isfile
from shutil import rmtree
from os import listdir, mkdir
import argparse

import h5py
from tqdm import tqdm
import random
import re
import pickle
from math import sqrt, acos, pi

import matplotlib.pyplot as plt

VERBOSE = False

def readH5(filename):
    f = h5py.File(filename, 'r')
    return f

# WRITE SEGMENTATION

def writeSegmentedPointCloudOBJ(out_filename, points, labels, colors = None):
    labels = labels.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    if VERBOSE:
        print('\nWriting Segmented Point Cloud...')
    for i in tqdm(range(N)) if VERBOSE else range(N):
        if colors is None:
            random.seed(labels[i])
            color = random.randint(0, 16000000)
            c = [0, 0, 0]
            c[0] = (color%255)
            color //= 255
            c[1] = (color%255)
            color //= 255
            c[2] = (color%255)
        else:
            c = colors[labels[i]]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0], points[i,1], points[i,2], c[0], c[1], c[2]))
    fout.close()

def createLabelsByType(h5_file):
    labels_type = np.zeros(shape = h5_file['gt_labels'].shape, dtype=int)
    soup_prog = re.compile('(.*)_soup_([0-9]+)$')
    type_to_indice = {
        'plane': 1,
        'cylinder': 2,
        'cone': 3,
        'sphere': 4,
    }
    if VERBOSE:
        print('\nCreating Labes by Type...')
    for key in tqdm(list(h5_file.keys())) if VERBOSE else list(h5_file.keys()):
        m = soup_prog.match(key)
        if m is not None:
            indices = h5_file[key]['gt_indices'][()]
            tp = pickle.loads(h5_file[key].attrs['meta'])['type']
            for i in indices:
                labels_type[i] = type_to_indice[tp]     
    return labels_type


# CALCULATE DISTANCES

def distance_points(A, B):
    AB = B - A
    return np.linalg.norm(AB, ord=2)

def angle_vectors(n1, n2):
    c = np.dot(n1, n2)/(np.linalg.norm(n1, ord=2)*np.linalg.norm(n2, ord=2))
    c = -1.0 if c < -1 else c
    c =  1.0 if c >  1 else c
    return acos(c)

def deviation_normals(n1, n2):
    a1 = angle_vectors(n1, n2)
    a2 = angle_vectors(n1, -n2)
    if abs(a1) < abs(a2):
        return a1
    return a2

def deviation_point_line(point, normal, curve):
    A = np.array(list(curve['location'])[0:3])
    v = np.array(list(curve['direction'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    #AP vector
    AP = P - A
    #equation to calculate distance between point and line using a direction vector
    return np.linalg.norm(np.cross(v, AP), ord=2)/np.linalg.norm(v, ord=2), abs(pi/2 - deviation_normals(v, n_p))

def deviation_point_circle(point, normal, curve):
    A = np.array(list(curve['location'])[0:3])
    n = np.array(list(curve['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    radius = curve['radius']

    AP = P - A
    #orthogonal distance between point and circle plane
    dist_point_plane = np.dot(AP, n)/np.linalg.norm(n, ord=2)

    #projection P in the circle plane and calculating the distance to the center
    P_p = P - dist_point_plane*n/np.linalg.norm(n, ord=2)
    dist_pointproj_center = np.linalg.norm(P_p - A, ord=2)
    #if point is outside the circle arc, the distance to the curve is used 
    a = dist_pointproj_center - radius
    b = np.linalg.norm(P - P_p, ord=2)

    #calculanting tangent vector to the circle in that point
    n_pp = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
    t = np.cross(n_pp, n)
    return sqrt(a**2 + b**2), abs(pi/2 - deviation_normals(t, n_p))

def deviation_point_sphere(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    radius = surface['radius']
    #normal of the point projected in the sphere surface
    AP = P - A
    n_pp = AP/np.linalg.norm(AP, ord=2)
    #simple, distance from point to the center minus the sphere radius
    #angle between normals
    return abs(distance_points(P, A) - radius), deviation_normals(n_pp, n_p)

def deviation_point_plane(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    n = np.array(list(surface['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    AP = P - A
    #orthogonal distance between point and plane
    #angle between normals
    angle = deviation_normals(n, n_p)
    if angle > pi/2:
        angle = pi - angle

    return abs(np.dot(AP, n)/np.linalg.norm(n, ord=2)), angle

def deviation_point_torus(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    n = np.array(list(surface['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    max_radius = surface['max_radius']
    min_radius = surface['min_radius']
    radius = (max_radius - min_radius)/2

    AP = P - A
    #orthogonal distance to the torus plane 
    h = np.dot(AP, n)/np.linalg.norm(n, ord = 2)

    line = surface
    line['direction'] = surface['z_axis']
    #orthogonal distance to the revolution axis line 
    d = deviation_point_line(point, normal, line)[0]

    #projecting the point in the torus plane
    P_p = P - h*n/np.linalg.norm(n, ord=2)
    #getting the direction vector, using center as origin, to the point projected
    v = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
    #calculating the center of circle in the direction of the input point
    B = (min_radius + radius)*v + A

    BP = P - B
    n_pp = BP/np.linalg.norm(BP, ord=2)

    return abs(distance_points(B, P) - radius), deviation_normals(n_pp, n_p)

def deviation_point_cylinder(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    radius = surface['radius']
    surface['direction'] = surface['z_axis']
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    #normal of the point projected in the sphere surface
    AP = P - A
    n_pp = AP/np.linalg.norm(AP, ord=2)
    #simple distance from point to the revolution axis line minus radius
    return abs(deviation_point_line(point, normal, surface)[0] - radius), deviation_normals(n_pp, n_p)

def deviation_point_cone(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    v = np.array(list(surface['z_axis'])[0:3])
    B = np.array(list(surface['apex'])[0:3])
    P = np.array(list(point)[0:3])
    radius = surface['radius']
    n_p = np.array(list(normal)[0:3])

    #height of cone
    h = distance_points(A, B)

    AP = P - A
    P_p = A + np.dot(AP, v)/np.dot(v, v) * v

    #distance from center of base to point projected
    dist_AP = distance_points(P_p, A)
    #distance from apex to the point projected
    dist_BP = distance_points(P_p, B)

    #if point is below the center of base, return the distance to the circle base line
    if dist_BP > dist_AP and dist_BP >= h:
        AP = P - A
        signal_dist_point_plane = np.dot(AP, v)/np.linalg.norm(v, ord=2)

        #projection P in the circle plane and calculating the distance to the center
        P_p = P - signal_dist_point_plane*v/np.linalg.norm(v, ord=2)
        dist_pointproj_center = np.linalg.norm(P_p - A, ord=2)
        #if point is outside the circle arc, the distance to the curve is used 
        if dist_pointproj_center > radius:
            #not using distance_point_circle function to not repeat operations
            a = dist_pointproj_center - radius
            b = np.linalg.norm(P - P_p, ord=2)
            n_pp = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
            t = np.cross(n_pp, v)
            return sqrt(a**2 + b**2), abs(pi/2 - deviation_normals(t, n_p))
        
        #if not, the orthogonal distance to the circle plane is used
        return abs(signal_dist_point_plane), deviation_normals(-v, n_p)
    #if point is above the apex, return the distance from point to apex
    elif dist_AP > dist_BP and dist_AP >= h:
        return distance_points(P, B), deviation_normals(v, n_p)

    #if not, calculate the radius of the circle in this point height 
    r = radius*dist_BP/h

    d = (P - P_p)/np.linalg.norm((P-P_p), ord=2)

    vr = r * d

    P_s = P_p + vr

    s = (P_s - B)/np.linalg.norm((P_s - B), ord=2)

    t = np.cross(d, v)

    n_pp = np.cross(t, s)

    #distance from point to the point projected in the revolution axis line minus the current radius
    return abs(distance_points(P, P_p) - r), deviation_normals(n_pp, n_p)

deviation_functions = {
    'line': deviation_point_line,
    'circle': deviation_point_circle,
    'sphere': deviation_point_sphere,
    'plane': deviation_point_plane,
    'torus': deviation_point_torus,
    'cylinder': deviation_point_cylinder,
    'cone': deviation_point_cone
}

def calculateError(h5_file):
    points = h5_file['gt_points']
    normals = h5_file['gt_normals']

    soup_prog = re.compile('(.*)_soup_([0-9]+)$')
    if VERBOSE:
        print('\nCalculating Deviation Errors...')
    keys = []
    for key in list(h5_file.keys()):
        m = soup_prog.match(key)
        if m is not None:
            keys.append(key)
    
    results = {}

    for i, key in tqdm(enumerate(keys)) if VERBOSE else enumerate(keys):
        indices = h5_file[key]['gt_indices'][()]
        data = pickle.loads(h5_file[key].attrs['meta'])
        tp = data['type']
        mean_distance = 0
        mean_normal = 0
        for i in indices:
            distance, normal_dev = deviation_functions[tp](points[i], normals[i], data)
            mean_distance += abs(distance)
            mean_normal += abs(normal_dev)
        if len(indices) > 0:
            mean_distance /= len(indices)
            mean_normal /= len(indices)

        if tp in results.keys():
            results[tp]['mean_distance'] = np.append(results[tp]['mean_distance'], mean_distance)
            results[tp]['mean_normal'] = np.append(results[tp]['mean_normal'], mean_normal)
        else:
            results[tp] = {}
            results[tp]['mean_distance'] = np.array([mean_distance])
            results[tp]['mean_normal'] = np.array([mean_normal])

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Geometric Primitive Fitting Results, works for dataset validation and for methods results')
    parser.add_argument('folder', type=str, help='dataset folder.')
    parser.add_argument('--h5_folder_name', type=str, default = 'h5', help='h5 folder name.')
    parser.add_argument('--result_folder_name', type=str, default = 'val', help='validation folder name.')
    parser.add_argument('-v', '--verbose', action='store_true', help='show more verbose logs.')
    parser.add_argument('-s', '--segmentation_gt', action='store_true', help='write segmentation ground truth.')
    parser.add_argument('-b', '--show_box_plot', action='store_true', help='show box plot of the data.')
    parser.add_argument('-md', '--max_distance_deviation', type=float, default=0.5, help='max distance deviation.')
    parser.add_argument('-mn', '--max_normal_deviation', type=float, default=10, help='max normal deviation.')

    args = vars(parser.parse_args())

    folder_name = args['folder']
    h5_folder_name = join(folder_name, args['h5_folder_name'])
    result_folder_name = join(folder_name, args['result_folder_name'])
    VERBOSE = args['verbose']
    write_segmentation_gt = args['segmentation_gt']
    box_plot = args['show_box_plot']
    max_distance_deviation = args['max_distance_deviation']
    max_normal_deviation = args['max_normal_deviation']*pi/180.
 
    if exists(h5_folder_name):
        h5_files = sorted([f for f in listdir(h5_folder_name) if isfile(join(h5_folder_name, f))])
    else:
        if VERBOSE:
            print('\nThere is no h5 folder.\n')
        exit()
    
    if exists(result_folder_name):
        rmtree(result_folder_name)
    mkdir(result_folder_name)

    seg_folder_name = join(result_folder_name, 'seg')

    if write_segmentation_gt:
        if exists(seg_folder_name):
            rmtree(seg_folder_name)
        mkdir(seg_folder_name)

    box_plot_folder_name = join(result_folder_name, 'boxplot')
    
    if box_plot:
        if exists(box_plot_folder_name):
            rmtree(box_plot_folder_name)
        mkdir(box_plot_folder_name)

    
    for h5_filename in h5_files if VERBOSE else tqdm(h5_files):
        if VERBOSE:
            print(f'\n-- Processing file {h5_filename}...')

        bar_position = h5_filename.rfind('/')
        point_position = h5_filename.rfind('.')
        base_filename = h5_filename[(bar_position+1):point_position]

        h5_file = readH5(join(h5_folder_name, h5_filename))

        if write_segmentation_gt:
            labels_type = createLabelsByType(h5_file)

            points = h5_file['gt_points']
            labels = h5_file['gt_labels']

            instances_filename = f'{base_filename}_instances.obj'
            writeSegmentedPointCloudOBJ(join(seg_folder_name, instances_filename), points, labels)

            types_filename = f'{base_filename}_types.obj'
            colors = [(255,255,255), (255,0,0), (0,255,0), (0,0,255), (255,255,0)]
            writeSegmentedPointCloudOBJ(join(seg_folder_name, types_filename), points, labels_type, colors=colors)
        
        error_results = calculateError(h5_file)

        number_of_primitives = 0
        distance_error = 0
        normal_dev_error = 0
        mean_distance = 0.
        mean_normal_dev = 0.
        for key in error_results.keys():
            number_of_primitives += len(error_results[key]['mean_distance'])
            distance_error += np.count_nonzero(error_results[key]['mean_distance'] > max_distance_deviation)
            normal_dev_error += np.count_nonzero(error_results[key]['mean_normal'] > max_normal_deviation)
            mean_distance += np.sum(error_results[key]['mean_distance'])
            mean_normal_dev += np.sum(error_results[key]['mean_normal'])
        
        distance_error /= number_of_primitives
        normal_dev_error /= number_of_primitives
        mean_distance /= number_of_primitives
        mean_normal_dev /= number_of_primitives
        
        if VERBOSE:
            print(f'{h5_filename} is processed.')

        print('\nTESTING REPORT:')
        print('\n- Total:')
        print('\t- Number of Primitives:', number_of_primitives)
        print('\t- Distance Error Rate:', (distance_error)*100, '%')
        print('\t- Normal Error Rate:', (normal_dev_error)*100, '%')
        print('\t- Mean Distance Error:', mean_distance)
        print('\t- Mean Normal Error:', mean_normal_dev)
        data_distance = []
        data_normal = []
        labels = []
        for key in error_results.keys():
            name = key[0].upper() + key[1:]
            number_of_primitives_loc = len(error_results[key]['mean_distance'])
            print(f'\n- {name}:')
            print('\t- Number of Primitives:', number_of_primitives_loc)
            print('\t- Distance Error Rate:', (np.count_nonzero(error_results[key]['mean_distance'] > max_distance_deviation)/number_of_primitives_loc)*100, '%')
            print('\t- Normal Error Rate:', (np.count_nonzero(error_results[key]['mean_normal'] > max_normal_deviation)/number_of_primitives_loc)*100, '%')
            print('\t- Mean Distance Error:', np.mean(error_results[key]['mean_distance']))
            print('\t- Mean Normal Error:', np.mean(error_results[key]['mean_normal']))
            data_distance.append(error_results[key]['mean_distance'])
            data_normal.append(error_results[key]['mean_normal'])
            labels.append(name)
        if box_plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.tight_layout(pad=2.0)
            ax1.set_title('Distance Deviation')
            ax1.boxplot(data_distance, labels=labels, autorange=False, meanline=True)
            ax2.set_title('Normal Deviation')
            ax2.boxplot(data_normal, labels=labels, autorange=False, meanline=True)
            plt.savefig(join(box_plot_folder_name, f'{base_filename}.png'))
            plt.show(block=False)
            plt.pause(10)
            plt.close()