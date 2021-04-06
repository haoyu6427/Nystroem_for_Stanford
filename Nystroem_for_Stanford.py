import numpy as np
import open3d as o3d
import random
from sklearn.kernel_approximation import Nystroem

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        \n'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

pcd = o3d.io.read_point_cloud("./bun_zipper_res3.ply")
a = np.asarray(pcd.points)
num_data = 20

for i in range(num_data):
    feature_map_nystroem = Nystroem(gamma=2,
                                     n_components=100)
    data_transformed = feature_map_nystroem.fit_transform(a)
    l = random.uniform(10.0,100.0)
    l = l*0.2
    print(i, l)
    random.shuffle(data_transformed)
    out = a + data_transformed[:,:3]/l
    output_file = 'bunny_def/'+str(i)+'.ply'
    NUM_POINT = a.shape[0]
    one = np.ones((NUM_POINT,1))*255
    zero = np.zeros((NUM_POINT,1))
    red = np.concatenate((one,zero,zero),1)
    green = np.concatenate((zero,one,zero),1)
    blue = np.concatenate((zero,zero,one),1)
    if(i%3==0):
        color = red
    if(i%3==1):
        color = green
    if(i%3==2):
        color = blue
    if(i==0):
        inputs= a
    else:
        inputs = out
    create_output(inputs, color, output_file)