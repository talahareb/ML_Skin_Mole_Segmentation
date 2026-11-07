import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mole_segmentation import resize_rgb, select_mole_cluster, segment_image

def test_resize_rgb_downscales():
    rgb=np.zeros((650,800,3),dtype=np.uint8)
    out=resize_rgb(rgb,200)
    assert max(out.shape[:2])==200

def test_cluster_prefers_interior():
    points=np.array([[0,0],[0,1],[1,0],[1,1],[4,4],[4,5],[5,4],[5,5]])
    labels=np.array([0,0,0,0,1,1,1,1])
    pts=select_mole_cluster(points,labels,(10,10),min_pixels=3)
    assert (pts[:,0]>=4).all()

def test_segment_generated_dark_circle():
    yy,xx=np.mgrid[:128,:128]
    r=((xx-64)**2+(yy-64)**2)**.5
    bg=np.clip(210+2*np.sin(xx/14),0,255)
    arr=np.repeat(bg[:,:,None],3,axis=2)
    arr[r<23]=45
    arr=np.asarray(arr,dtype=np.uint8)
    result=segment_image(arr,max_side=128,eps=3,min_samples=4,median_size=3)
    assert result.valid
    assert result.mask[64,64]
    assert not result.mask[0,0]
    assert result.edge.any()

def test_invalid_inputs():
    with pytest.raises(ValueError): resize_rgb(np.zeros((20,20,3),dtype=np.uint8),max_side=12)
