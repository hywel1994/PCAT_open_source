import h5py
import numpy as np
import os
import math
import cv2

LASER_ANGLES = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
DISTANCE_RESOLUTION = 0.002
ROTATION_RESOLUTION = 0.01
ROTATION_MAX_UNITS = 36000

def calc(dis, azimuth, laser_id):
    R = dis * DISTANCE_RESOLUTION
    omega = LASER_ANGLES[laser_id] * np.pi / 180.0
    alpha = azimuth * ROTATION_RESOLUTION * np.pi / 180.0
    X = R * np.cos(omega) * np.sin(alpha)
    Y = R * np.cos(omega) * np.cos(alpha)
    Z = R * np.sin(omega)
    return [X, Y, Z, 0]

def write_pcb(points, path, WIDTH, i):
    
    path = path+'/training/pcb/'+str(i).zfill(6)+'.pcd'
    if os.path.exists(path):
        os.remove(path)

    out = open(path, 'a')
    # headers
    out.write('# .PCD v.7 - Point Cloud Data file format\nVERSION .7')
    out.write('\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(WIDTH)
    out.write(string)
    out.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(WIDTH)
    out.write(string)
    out.write('\nDATA ascii')

    #datas
    for i in range(WIDTH):
        string = '\n' + str(points[i][0]) + ' ' + str(points[i][1]) + ' ' \
                +str(points[i][2]) + ' ' + str(points[i][3])
        out.write(string)
    out.close()

root_dir = "/home/hywel/DataDisk/lidar-image-gps-ahrs"
for filename in os.listdir(root_dir):
    print (filename)
    path_dir = os.path.join(root_dir, filename)
    for name in ['calib', 'image_2', 'label_2', 'velodyne', 'pcb']:
        tmp = os.path.join(path_dir,'training',name)
        if not os.path.exists(tmp):
            os.makedirs(tmp)
    
    image_file_name = os.path.join(root_dir, filename, filename+'.avi')
    h5_file_name = os.path.join(root_dir, filename, filename+'.h5')
    
    try:
        f = h5py.File(h5_file_name, 'r')
    except:
        print ("can't open ", h5_file_name)
        continue

    lidar_data = f['lidar_data']
    extra_data = f['extra_data']
    timestamp = f['timestamp']
    
    cap = cv2.VideoCapture(image_file_name)
    # frames = []
    # while(1):
    #     # get a frame
    #     ret, frame = cap.read()
    #     if ret:
    #         frames.append(frame)
    #     else:
    #         break
    # print ("image frames: ",len(frames))


    num_old = -1
    for i, num in enumerate(range(0,len(lidar_data),100)):
        print ("record frame: ", num)

        for _ in range(num_old,num):
            ret, frame = cap.read()
            if not ret:
                print ("lack vidio frame")
                break
        num_old = num

        points = []
        one_lidar = lidar_data[num,:]
        assert len(one_lidar)//75==408
        WIDTH = 75*24*16
        for j in range(75):
            tmp_data = one_lidar[408*j:408*(j+1)]
            for k in range(24):
                dis_list = tmp_data[16*k:16*(k+1)]
                azimuth = tmp_data[384+k]
                for w in range(16):
                    points.append(calc(dis_list[w], azimuth, w))

        write_pcb(points, path_dir, WIDTH, i)
        name = path_dir + "/training/image_2/"+str(i).zfill(6)+".png"
        cv2.imwrite(name, frame)

