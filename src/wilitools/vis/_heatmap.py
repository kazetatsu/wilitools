from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get_heatmap(x:np.ndarray, img_size:tuple[int,int]=(960,540)) -> Image.Image:
    # Remove unnecessary parts
    fig = plt.figure(dpi=1, figsize=img_size)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot()
    ax.axis('off')

    ax.pcolor(x) # Draw heatmap

    # Convert: matplotlib figure -> Pillow image
    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    return img
