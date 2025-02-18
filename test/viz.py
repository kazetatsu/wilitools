import numpy as np
from PIL import Image
from wilitools import *
from wilitools.vis import *

def test_heatmap():
    x = np.array([[4, 8, 12, 16], [2, 4, 6, 8], [0, 0, 0, 0]], dtype=np.float32)
    img = get_heatmap(x)
    assert isinstance(img, Image.Image)


def test_overview():
    floor = Floor(-6.0, 6.0, -5.0, 5.0)
    avrs = np.array([[4,0], [-2,3], [-2,-3]], dtype=np.float32)
    tr_probs = np.array([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1], [0.4, 0.3, 0.3]], dtype=np.float32)
    img = get_overview(floor, avrs, tr_probs)
    assert isinstance(img, Image.Image)
