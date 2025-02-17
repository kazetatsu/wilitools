import numpy as np
from PIL import Image
from wilitools.vis import *

def test_heatmap():
    x = np.array([[4, 8, 12, 16], [2, 4, 6, 8], [0, 0, 0, 0]], dtype=np.float32)
    img = get_heatmap(x)
    assert isinstance(img, Image.Image)
