#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from math import ceil
from struct import pack, unpack

import argparse

import sys

POSSIBLE_FIELDS = ['x', 'y', 'z', 'i', 'rgb', 'rgba', 'normal_x', 'normal_y', 'normal_z', 'curvature', 'label']

def mount_pcd_fields(in_fields):
    pcd_fields = []
    for f in POSSIBLE_FIELDS:
        comp_c = 0
        for i_f in in_fields:
            if f == 'rgb' or f == 'rgba':
                if i_f in f: 
                    comp_c += 1
            if f == i_f:
                pcd_fields.append(f)
                comp_c = 0
                break
        if comp_c == len(f):
            pcd_fields.append(f)
    
    if 'rgb' in pcd_fields and 'rgba' in pcd_fields:
        pcd_fields.remove('rgb')
    return pcd_fields
        
def mount_header(n_points, pcd_fields, binary=False):
    header_f = ''
    header_s = ''
    header_t = ''
    for f in pcd_fields:
        header_f+= f + ' '
        header_s+= '4 '
        if f != 'label':
            header_t+= 'F '
        else:
            header_t+= 'U '
    header_f = header_f[0:-1]
    header_s = header_s[0:-1]
    header_t = header_t[0:-1]

    size = len(header_s.split())

    header_c = '1 '*size
    header_c = header_c[0:-1]

    file_type = 'binary' if binary else 'ascii'

    header = 'VERSION .7\nFIELDS {}\nSIZE {}\nTYPE {}\nCOUNT {}\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA {}\n'.format(header_f, header_s, header_t, header_c, n_points, n_points, file_type)

    return header

def mount_ascii_data_line(line_s, in_fields, pcd_fields):
    line = ''
    next_index = 0
    for f in pcd_fields:
        if 'rgb' == f or 'rgba' == f:
            r = np.uint8(line_s[in_fields.index('r')])
            g = np.uint8(line_s[in_fields.index('g')])
            b = np.uint8(line_s[in_fields.index('b')])
            data = np.uint32(np.uint32(r) << 16 | np.uint32(g) << 8 | np.uint32(b))
            if 'rgba' == f:
                a = np.uint8(in_fields.index('a'))
                data = np.uint32(np.uint32(a) << 24 | data)
            data_p = pack('I', data)
            line += str(unpack('f', data_p)[0]) + ' '
        else:
            if f == 'i':
                intensity = int(line_s[in_fields.index(f)])
                intensity/= 255
                line_s[in_fields.index(f)] = str(intensity)
                
            line += line_s[in_fields.index(f)] + ' '
            
    line = line[0:-1]
    line += '\n'
    return line

def mount_binary_data_line(line_s, in_fields, pcd_fields):
    line = []
    for f in pcd_fields:
        if 'rgb' == f or 'rgba' == f:
            r = np.uint8(line_s[in_fields.index('r')])
            g = np.uint8(line_s[in_fields.index('g')])
            b = np.uint8(line_s[in_fields.index('b')])
            data = np.uint32(np.uint32(r) << 16 | np.uint32(g) << 8 | np.uint32(b))
            if 'rgba' == f:
                a = np.uint8(in_fields.index('a'))
                data = np.uint32(np.uint32(a) << 24 | data)
            data_p = pack('I', data)
            line += [unpack('f', data_p)[0]]
        else:
            if f == 'i':
                intensity = int(line_s[in_fields.index(f)])
                intensity/= 255
                line_s[in_fields.index(f)] = intensity
                
            line += [float(line_s[in_fields.index(f)])]
    return line

def mount_data_line(line_s, in_fields, pcd_fields, binary=False):
    if binary:
        return mount_binary_data_line(line_s, in_fields, pcd_fields)
    else:
        return mount_ascii_data_line(line_s, in_fields, pcd_fields)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts simple textual point cloud formats (TXT, XYZ, PTS) to PCD')
    parser.add_argument('input', type=str, help='input file in .txt, .xyz or .pts.')
    parser.add_argument('output', type=str, help='output file in .pcd.')
    parser.add_argument('--fields', type=str, default = 'x,y,z,a,r,g,b', help='fields in order contained in textual file. Possible fields are:' + ','.join(POSSIBLE_FIELDS) + '. Put \'_\' in the fields to ignore. Default = x,y,z,a,r,g,b')
    parser.add_argument('--buffer_size', type = int, default = 10000000, help='size of buffer to be used. It is needed to not use all the RAM from computer. Default = 10000000')
    parser.add_argument('--number_of_points', type = int, default = 0, help='number of points in the file. It speeds-up the process when the file is TXT or XYZ. Default = 0')
    args = vars(parser.parse_args())

    inputname = args['input']
    outputname = args['output']
    in_fields = args['fields'].split(',')
    pcd_fields = mount_pcd_fields(in_fields)
    buffer_size = args['buffer_size']
    number_of_points = args['number_of_points']

    f1 = open(inputname, 'r')
    f2 = open(outputname, 'wb')
    if inputname[-4:] == '.txt' or inputname[-4:] == '.xyz':
        print('Processing a {} file...'.format(inputname[-4:]))
        print('Getting number of points...')
        number_of_points = 0
        for line in tqdm(f1):
            number_of_points += 1
        f1 = open(inputname, 'r')
    elif inputname[-4:] == '.pts':
        print('Processing a {} file...'.format(inputname[-4:]))
        print('Getting number of points...')
        number_of_points = int(f1.readline())
    else:
        print('This file type cannot be processed.')
    
    print('Done. {} points to be processed.'.format(number_of_points))
    
    n = 0
    buffer_array = []
    i = 0
    b = 0
    header = mount_header(number_of_points, pcd_fields, binary=True)
    f2.write(bytes(header.encode('ascii')))
    for line in tqdm(f1):
        if i == buffer_size:
            i = 0
            f2.write(bytes(pack("%sf" % len(buffer_array), *buffer_array)))
            buffer_array = []
            b+=1
        line_s = line.strip().split(' ')
        if len(line_s) != len(in_fields):
            print('\nFields has {} dimensions, but the lines of the file has {}.'.format(len(in_fields), len(line_s)))
            exit()
        l = mount_data_line(line_s, in_fields, pcd_fields, binary=True)
        buffer_array += l
        n += 1
        i += 1
    if i > 1:
        f2.write(bytes(pack("%sf" % len(buffer_array), *buffer_array)))
    print('Done. {} points were processed.'.format(n))
    f2.close()
