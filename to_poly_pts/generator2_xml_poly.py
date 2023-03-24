# coding=utf-8
import glob, os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import *
from PIL import Image
import time
import re 
import xml.etree.cElementTree as ET



def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image

def replace_all(s, old, new, reg = False):
    if reg:
        import re
        targets = re.findall(old, s)
        for t in targets:
            s = s.replace(t, new)
    else:
        s = s.replace(old, new)
    return s


def remove_all(s, sub):
    return replace_all(s, sub, '')


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))

def split_edge_seqence(points, n_parts):
    pts_num = points.shape[0]
    long_edge = [(i, (i + 1) % pts_num) for i in range(pts_num-1)]
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part
        while cur_end > point_cumsum[cur_node + 1]:
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / max(edge_length[cur_node], 1)
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


def generate_ctr_point(points, num=16):
    n_disk = num // 2 - 1
    pn = points.shape[0]//2
    top = split_edge_seqence(points[:pn], n_disk)
    bot = split_edge_seqence(points[pn:], n_disk)
    points = np.concatenate([np.array(top), np.array(bot)], axis=0)
    return points

# test
root  = "./CTW1500/"
labels = glob.glob(root+'labels/*.xml')
labels.sort()

if not os.path.isdir(root+'poly_gen_labels'):
            os.mkdir(root+'poly_gen_labels')

max_word_len = 0
max_word_str = ""
for il, label in enumerate(labels):
    print('Processing: '+label)
    imgdir = label.replace('labels', 'images').replace('.xml', '.jpg')

    outgt = open(label.replace('labels', 'poly_gen_labels').replace('.xml', '.txt'), 'w')

    data = []
    cts  = []
    centers = []


    root_xml = ET.parse(label).getroot()
    for tag in root_xml.findall('image/box'):
        ct = tag.find("label").text.strip()
        gt = list(map(int, tag.find("segs").text.strip().split(",")))
        pts = []
        for pt in tag.findall('pts'):
            pp = pt.attrib
            pts.append([int(pp["x"]), int(pp["y"])])
        pts = np.array(pts).astype(np.int32)
        coords = np.stack([gt[0::2], gt[1::2]]).T.astype(np.int32)

        if len(ct) > max_word_len:
            max_word_str = ct

        max_word_len = max(max_word_len, len(ct))

        if len(ct) > 100:
            print(ct)
            pn = coords.shape[0]
            cn = len(ct)
            coords1 =np.concatenate([coords[0:4], coords[10:14]], axis=0)
            pts1 =  pts[:cn//2]
            ct１ = ct[:cn // 2]
            print(ct１)

            ctr_pts = generate_ctr_point(coords1, num=16)
            data.append(ctr_pts.astype(np.int32))
            cts.append(ct１)
            centers.append(pts1)

            coords = coords[3:11]
            pts = pts[cn // 2:]
            ct = ct[cn // 2 :]
            print(ct)

        ctr_pts = generate_ctr_point(coords, num=16)
        data.append(ctr_pts.astype(np.int32))
        cts.append(label)
        centers.append(pts)

    ############## top
    ############## top
    # img = cv2.imread(imgdir)
    # img = pil_load_img(imgdir)
    img = Image.open(imgdir).convert("RGB")
    img = np.ascontiguousarray(np.array(img)[:, :, ::-1])
    # img = plt.imread(imgdir)
    
    for iid, ddata in enumerate(data):
        cv2.drawContours(img, [ddata], -1, (0, 255, 0), 1)
        for j, pp in enumerate(ddata[:]):
            if j == 0:
                cv2.circle(img, (int(pp[0]), int(pp[1])), 3, (0, 0, 255), -1)
            elif j == 1:
                cv2.circle(img, (int(pp[0]), int(pp[1])), 3, (255, 0, 255), -1)
            else:
                cv2.circle(img, (int(pp[0]), int(pp[1])), 3, (0, 255, 255), -1)

        outstr = ""
        for pp in ddata:
            outstr += "{},{},".format(pp[0], pp[1])

        outstr += "||||,"
        for pp in centers[iid]:
            outstr += "{},{},".format(pp[0], pp[1])
        outstr += "||||{}\n".format(cts[iid])

        outgt.writelines(outstr)
    outgt.close()

    if not os.path.isdir(root + 'img_vis'):
        os.mkdir(root + 'img_vis')
    cv2.imwrite(root + 'img_vis/' + os.path.basename(imgdir), img)

print(max_word_len)
print(max_word_str)
