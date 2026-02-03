import os, cv2, shutil, random
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from skimage import morphology, measure

# ---------- 参数 ----------
slice_dir  = 'slice_dir'          # 原始二维切片目录
mask_dir   = 'mask_dir'           # 生成的掩膜保存目录
dataset_dir= 'dataset'            # 最终划分目录
K          = 2                    # 聚类中心数
random.seed(42)                   # 复现用

os.makedirs(mask_dir,    exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)
for sub in ['train','val','test']:
    os.makedirs(os.path.join(dataset_dir,sub,'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir,sub,'masks'),  exist_ok=True)

# ---------- 1. 预处理 + K-means 分割 ----------
def process_one_slice(img_path):
    """返回二值掩膜（0/1）"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 确保灰度
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)  # 归一化到0-255
    h, w = img.shape
    vec = img.reshape(-1, 1)

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vec).reshape(h, w)

    # 肺实质灰度低，选均值更小的簇
    cluster_mean = [img[labels == i].mean() for i in range(K)]
    lung_cluster = np.argmin(cluster_mean)
    binary = (labels == lung_cluster).astype(np.uint8)

    # 形态学开闭运算
    binary = morphology.opening(binary, morphology.disk(3))
    binary = morphology.closing(binary, morphology.disk(3))

    # 保留最大连通区域
    labels_m = measure.label(binary, connectivity=2)
    props  = measure.regionprops(labels_m)
    if not props:           # 极端情况无区域
        return np.zeros_like(binary)
    largest = max(props, key=lambda x: x.area)
    final_mask = (labels_m == largest.label).astype(np.uint8)
    return final_mask


# ---------- 2. 批量生成掩膜 ----------
slice_list = sorted(glob(os.path.join(slice_dir, '*')))
for sp in slice_list:
    mask = process_one_slice(sp)
    save_path = os.path.join(mask_dir, os.path.basename(sp).rsplit('.',1)[0]+'.jpg')
    cv2.imwrite(save_path, mask*255)   # 保存为0-255 JPG

# ---------- 3. 8:1:1 随机划分 ----------
all_pairs = list(zip(sorted(glob(os.path.join(slice_dir, '*'))),
                     sorted(glob(os.path.join(mask_dir,   '*')))))
random.shuffle(all_pairs)
n = len(all_pairs)
train_split, val_split = int(0.8*n), int(0.9*n)

splits = {'train': all_pairs[:train_split],
          'val':   all_pairs[train_split:val_split],
          'test':  all_pairs[val_split:]}

for phase, pairs in splits.items():
    for img_path, mask_path in pairs:
        shutil.copy(img_path,
                    os.path.join(dataset_dir, phase, 'images', os.path.basename(img_path)))
        shutil.copy(mask_path,
                    os.path.join(dataset_dir, phase, 'masks',  os.path.basename(mask_path)))

print('Done. 掩膜与划分结果已保存。')