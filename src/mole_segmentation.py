"""Classical image-segmentation pipeline for coursework demonstrations.

Image processing: grayscale -> K-Means -> spatial DBSCAN -> median -> Sobel.
Outputs indicate candidate dark regions, *not* melanoma diagnosis or risk.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import DBSCAN, KMeans


@dataclass
class SegmentationResult:
    image: np.ndarray
    grayscale: np.ndarray
    quantized: np.ndarray
    candidate_mask: np.ndarray
    mask: np.ndarray
    edge: np.ndarray
    overlay: np.ndarray
    selected_pixels: int
    area_fraction: float
    valid: bool
    message: str


def resize_rgb(image: np.ndarray, max_side: int = 320) -> np.ndarray:
    """Downscale for reasonable CPU/memory use in pixel-coordinate DBSCAN."""
    if max_side < 64:
        raise ValueError('max_side must be at least 64 pixels')
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError('Expected RGB/RGBA image')
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr[..., :3], 'RGB')
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(im.convert('RGB')).copy()


def select_mole_cluster(points: np.ndarray, labels: np.ndarray, shape: tuple[int, int], min_pixels: int = 40) -> np.ndarray:
    """Prefer substantial, interior clusters; fall back to largest non-noise cluster."""
    height, width = shape
    candidates = []
    for label in np.unique(labels):
        if label == -1:
            continue
        pts = points[labels == label]
        area = pts.shape[0]
        if area < min_pixels:
            continue
        touches = (pts[:,0].min() <= 1 or pts[:,0].max() >= height-2 or
                   pts[:,1].min() <= 1 or pts[:,1].max() >= width-2)
        center = pts.mean(axis=0)
        centered = np.linalg.norm((center - [height/2, width/2]) / [height, width])
        # Largest spatially connected dark region that avoids border artifacts.
        candidates.append((bool(touches), -area, centered, pts))
    if not candidates:
        return np.empty((0,2),dtype=np.int32)
    candidates.sort(key=lambda v:(v[0],v[1],v[2]))
    return candidates[0][3]


def segment_image(image: np.ndarray, *, max_side: int = 320,
                  n_clusters: int = 3, eps: float = 6.0,
                  min_samples: int = 12, median_size: int = 5,
                  random_state: int = 0) -> SegmentationResult:
    """Segment a candidate lesion; parameters are heuristic and image-dependent."""
    if not 2 <= n_clusters <= 8:
        raise ValueError('n_clusters must be between 2 and 8')
    if eps <= 0 or min_samples <= 0 or median_size < 1:
        raise ValueError('eps, min_samples, and median_size must be positive')
    rgb = resize_rgb(image,max_side=max_side)
    h,w=rgb.shape[:2]
    gray=np.mean(rgb,axis=2).astype('uint8')
    values=gray.reshape(-1,1).astype('float32')
    if len(np.unique(values)) < n_clusters:
        raise ValueError('Image has too few grayscale intensity levels')
    # Sample pixels to fit a lightweight KMeans; classify every pixel by centroid.
    rng = np.random.default_rng(random_state)
    idx = rng.choice(values.shape[0],size=min(12000,len(values)),replace=False)
    model = KMeans(n_clusters=n_clusters,random_state=random_state,n_init=5)
    model.fit(values[idx])
    centers=model.cluster_centers_.ravel()
    labels=np.argmin(abs(values - centers[None,:]),axis=1).reshape(h,w)
    quant=np.clip(centers[labels],0,255).astype('uint8')
    counts=np.bincount(labels.ravel(),minlength=n_clusters)
    sorted_indices=np.argsort(centers)
    # Match original coursework: darkest intensity cluster, avoiding broad background.
    dark_index=next((int(k) for k in sorted_indices if counts[k]<0.4*h*w),int(sorted_indices[0]))
    candidate=(labels==dark_index)
    points=np.argwhere(candidate)
    mask=np.zeros((h,w),dtype=bool)
    msg='Candidate lesion region identified'
    if len(points)>=min_samples:
        dbscan=DBSCAN(eps=eps,min_samples=min_samples,metric='euclidean').fit(points)
        chosen=select_mole_cluster(points,dbscan.labels_,(h,w))
        if len(chosen):
            mask[chosen[:,0],chosen[:,1]]=True
        else:
            msg='No substantial non-noise DBSCAN component found'
    else:
        msg='Too few candidate dark pixels'
    if mask.any():
        # Original workflow uses binary-image median smoothing before Sobel.
        mask=ndimage.median_filter(mask.astype('uint8'),size=median_size)>0
    smooth=mask.astype(float)*255
    gx=ndimage.sobel(smooth,axis=1,mode='constant')
    gy=ndimage.sobel(smooth,axis=0,mode='constant')
    grad=np.hypot(gx,gy)
    edge=(grad>0.3*grad.max()) if grad.max()>0 else np.zeros((h,w),dtype=bool)
    overlay=rgb.copy()
    overlay[edge]=np.array([255,75,60],dtype='uint8')
    total=int(mask.sum())
    return SegmentationResult(rgb,gray,quant,candidate,mask,edge,overlay,
                             selected_pixels=total,area_fraction=total/(h*w),
                             valid=bool(total and edge.any()),message=msg)


def segment_file(path: str|Path, **kwargs) -> SegmentationResult:
    with Image.open(path) as image:
        rgb=np.asarray(image.convert('RGB'))
    return segment_image(rgb,**kwargs)


def write_panels(result: SegmentationResult, output_path: str|Path, title: str='Image segmentation') -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[(result.image,'Original'),(result.grayscale,'Grayscale'),
            (result.quantized,'K-Means intensity clusters'),
            (result.candidate_mask,'Dark-cluster pixels'),
            (result.mask,'DBSCAN + median mask'),
            (result.overlay,'Sobel boundary overlay')]
    fig, axes=plt.subplots(2,3,figsize=(12,7.8))
    for ax,(im,label) in zip(axes.flat,panels):
        ax.imshow(im,cmap='gray' if im.ndim==2 else None,vmin=0 if im.ndim==2 else None,vmax=255 if im.dtype==np.uint8 and im.ndim==2 else None)
        ax.set_title(label,fontsize=12);ax.axis('off')
    fig.suptitle(title,fontsize=16,y=.99)
    fig.tight_layout(rect=(0,0,1,.96))
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=150,bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description='Classical skin-lesion segmentation (research demonstration only)')
    parser.add_argument('--input',type=Path,help='One RGB image')
    parser.add_argument('--input-dir',type=Path,help='Folder with JPG/PNG files')
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
    parser.add_argument('--max-side',type=int,default=320)
    parser.add_argument('--eps',type=float,default=6.)
    parser.add_argument('--min-samples',type=int,default=12)
    parser.add_argument('--median-size',type=int,default=5)
    args=parser.parse_args()
    if bool(args.input)==bool(args.input_dir):
        parser.error('Specify exactly one of --input or --input-dir')
    files=[args.input] if args.input else sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    success=0
    for file in files:
        try:
            result=segment_file(file,max_side=args.max_side,eps=args.eps,min_samples=args.min_samples,median_size=args.median_size)
            write_panels(result,args.output_dir/(file.stem+'_panels.png'),title=file.name)
            Image.fromarray(result.overlay).save(args.output_dir/(file.stem+'_overlay.png'))
            success+=int(result.valid)
            print(f'{file.name}: {result.message}; valid={result.valid}, selected_pixels={result.selected_pixels}')
        except Exception as exc:
            print(f'{file.name}: error {exc}')
    print(f'Processed {len(files)} image(s); {success} generated non-empty mask + border (not clinical validation).')

if __name__=='__main__':
    main()
