# -*- coding: utf-8 -*-
# SUBSTITUTE np.nan VALUES IN THIS FILE
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import os
import sys
#%%
folderpath = './moles/'
files = os.listdir(folderpath)
#%%
plotfig = True
np.set_printoptions(precision=2)# use only two decimal digits when printing numbers
plt.close('all')# close previously opened pictures

def process_one_image(filein,
                      epsilon=10.0,
                      M=20,
                      margin=8,
                      delta=3,
                      plotfig=False):
    print("\nProcessing:", filein)
    im_or = mpimg.imread(os.path.join(folderpath, filein))

    gray_image = np.mean(im_or, axis=2).astype(np.uint8)
    N1, N2 = gray_image.shape

    # ===== K-Means 量化 =====
    Ncluster = 3
    kmeans = KMeans(n_clusters=Ncluster, random_state=0)
    im_1D = gray_image.reshape((N1 * N2, 1))
    kmeans.fit(im_1D)
    Ncluster = len(kmeans.cluster_centers_)
    kmeans_centroids = kmeans.cluster_centers_.astype('uint8')

    im_1D_quant = im_1D.copy()
    for kc in range(Ncluster):
        im_1D_quant[(kmeans.labels_ == kc), :] = kmeans_centroids[kc, :]
    im_quant = im_1D_quant.reshape((N1, N2))

    if plotfig:
        plt.figure()
        plt.imshow(im_or, interpolation='none')
        plt.title(f'original image: {filein}')
        plt.figure()
        plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255, interpolation='none')
        plt.title('gray image')
        plt.figure()
        plt.imshow(im_quant, cmap='gray', interpolation='none')
        plt.title('image with quantized colors (after K-Means)')

    # ===== 找最暗簇、得到 mole_pos =====
    centroids = kmeans_centroids
    # 找最暗簇，但要排除像素数太多的簇（超过画面的 40%）    
    counts = [np.sum(kmeans.labels_ == i) for i in range(Ncluster)]
    sorted_idx = np.argsort(centroids.flatten())   # 从最暗到最亮排序

    i_col = None
    total_pixels = N1 * N2
    for idx in sorted_idx:
        if counts[idx] < 0.4 * total_pixels:   # 避免整体背景被选中
            i_col = idx
            break

    if i_col is None:
        print("!! Couldn't find valid darkest cluster.")
        return False

    im_clust = kmeans.labels_.reshape(N1, N2)
    mole_pos = np.argwhere(im_clust == i_col)

    # ===== DBSCAN =====
    clusters = DBSCAN(eps=epsilon, min_samples=M,
                      metric='euclidean').fit(mole_pos)
    id_clusters, count_id_clusters = np.unique(clusters.labels_, return_counts=True)
    print('Number of points in each cluster:', count_id_clusters)
    print('Indexes of clusters:', id_clusters)

    # 去掉噪声 -1
    valid_labels = id_clusters[id_clusters != -1]
    valid_counts = count_id_clusters[id_clusters != -1]
    if len(valid_labels) == 0:
        print("!! No valid clusters found, mark as FAIL")
        return False  # 这张图失败

    i_mole = valid_labels[np.argmax(valid_counts)]
    true_mole_pos = mole_pos[clusters.labels_ == i_mole]

    # 简单的“成功/失败”判定：像素太少也算失败
    if true_mole_pos.shape[0] < 50:
        print("!! Mole cluster too small, mark as FAIL")
        return False

    im_only_mole_gray = 0 * gray_image - 1
    x = true_mole_pos[:, 0]
    y = true_mole_pos[:, 1]
    im_only_mole_gray[x, y] = gray_image[x, y]

    im_mole_pos = np.ones((N1, N2), dtype='uint8') * 255
    im_mole_pos[x, y] = 0

    # ===== 裁剪 =====
    min_x = np.min(true_mole_pos[:, 0]) - margin
    max_x = np.max(true_mole_pos[:, 0]) + margin
    min_y = np.min(true_mole_pos[:, 1]) - margin
    max_y = np.max(true_mole_pos[:, 1]) + margin

    im_cropped_col = im_or[min_x:max_x, min_y:max_y, :]
    im_cropped_mole_pos = im_mole_pos[min_x:max_x, min_y:max_y]

    if plotfig:
        plt.figure()
        plt.imshow(im_cropped_mole_pos, cmap='gray', interpolation='none')
        plt.title('cropped image, mole position')

    # ===== 平滑（中值滤波）=====
    N1c, N2c = im_cropped_mole_pos.shape
    im_cropped_mole_pos_filt = 0 * im_cropped_mole_pos + 255
    for kr in range(delta, N1c - delta):
        for kc in range(delta, N2c - delta):
            sub = im_cropped_mole_pos[kr - delta:kr + delta,
                                      kc - delta:kc + delta]
            im_cropped_mole_pos_filt[kr, kc] = np.median(sub)

    # ===== Sobel =====
    kern1 = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
    kern2 = kern1.T

    Gx = np.zeros((N1c, N2c), dtype=float)
    Gy = np.zeros((N1c, N2c), dtype=float)
    img = im_cropped_mole_pos_filt.astype(float)

    for r in range(1, N1c - 1):
        for c in range(1, N2c - 1):
            patch = img[r-1:r+2, c-1:c+2]
            Gx[r, c] = np.sum(patch * kern1)
            Gy[r, c] = np.sum(patch * kern2)

    G = np.sqrt(Gx**2 + Gy**2)
    if np.all(G == 0):
        print("!! Sobel produced all zero gradient -> FAIL")
        return False
    G = G / (G.max() + 1e-8) * 255
    th = 0.3 * 255
    border = np.zeros_like(im_cropped_mole_pos_filt, dtype='uint8')
    border[G > th] = 255

    if plotfig:
        plt.figure()
        plt.imshow(im_cropped_col, interpolation='none')
        plt.imshow(border, cmap='gray', interpolation='none', alpha=0.2)
        plt.title(f'border on cropped image: {filein}')
        plt.show()

    # 最后如果一切正常，返回 True
    return True

if __name__ == '__main__':
    # 默认参数（适合大部分图片）
    default_params = dict(epsilon=10.0, M=20, margin=8, delta=3)

    # 针对特别难的几张图，可以在这里单独设参数（先留空，之后慢慢加）
    special_params = {
        # 'high_risk_3.jpg': dict(epsilon=5.0, M=5),
        # 'medium_risk_7.jpg': dict(epsilon=5.0, M=4, delta=4),
    }

    bad_images = []

    # 遍历所有 jpg 图片，不画图，只看成功/失败
    for fname in sorted(files):
        if not fname.endswith('.jpg'):
            continue

        params = default_params.copy()
        if fname in special_params:
            params.update(special_params[fname])

        ok = process_one_image(fname, plotfig=False, **params)
        if not ok:
            bad_images.append(fname)

    print("\n====== SUMMARY ======")
    print("Total images:", len([f for f in files if f.endswith('.jpg')]))
    print("Failed images:", bad_images)
