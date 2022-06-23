# from concurrent.futures import as_completed
# from multiprocessing import Pool
# from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

from concurrent.futures import as_completed

import os
from unittest import result
import numpy as np
import cProfile
import cv2
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

SCALES = [1080, 1080]
images = ['/Users/yangxiaorui/Desktop/0.png',
          '/Users/yangxiaorui/Desktop/1.png',
          '/Users/yangxiaorui/Desktop/2.png',
          '/Users/yangxiaorui/Desktop/3.png',
          '/Users/yangxiaorui/Desktop/4.png']

images = ['0.png',
          '1.png',
          '2.png',
          '3.png',
          '4.png']

idxs = [0, 1, 2, 3, 4]

def blur_p(idx):
    image = images[idx]
    img = cv2.imread(image)
    im_shape = img.shape
    scales = SCALES
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]

    factor = 3.0
    (h, w) = img.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    if kW % 2 == 0:
        kW -= 1
    if kH % 2 == 0:
        kH -= 1

    for i in range(10):
        box = np.array([600, 700, 800, 900])
        face_image = img[box[1]:box[3], box[0]:box[2]]
        face_image = cv2.GaussianBlur(face_image, (kW, kH), 0)
        img[box[1] : box[3], box[0] : box[2]] = face_image
    # cv2.imwrite(image.replace(".png", "_b.jpg"), img)

# def main2(iterations=10):
#     for _ in tqdm(range(iterations)):
#         futures = []
#         with Pool(processes=4) as pool:
#             for idx in idxs:
#                 futures.append(pool.apply_async(blur_p, [idx]))
#             for idx, future in enumerate(futures):
#                 future.get()
    

def main():
    iterations = 10
    futures = []

    for _ in tqdm(range(iterations)):
        with ThreadPoolExecutor(max_workers=16) as pool:
            for idx in idxs:
                futures.append(pool.submit(blur_p, idx))
        
        # results = []
        # for res in as_completed(futures):
        #     results.append(res.result())

    


# cProfile.run("main()")

def main_old(iterations=10):
    for _ in range(iterations):
        for idx in tqdm(idxs):
            blur_p(idx)

# cProfile.run("main()")
main()
main_old()