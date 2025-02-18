import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .._floor import Floor


def get_overview(
    floor:Floor, avrs:np.ndarray, tr_prob:np.ndarray,
    img_size:tuple[int,int]=(960,540)
) -> Image.Image:
    n = tr_prob.shape[0]

    centers = avrs - np.array((floor.x_min, floor.y_max))
    centers *= np.array([
        float(img_size[0]) / (floor.x_max - floor.x_min),
        -float(img_size[1]) / (floor.y_max - floor.y_min)
    ])
    centers = centers.astype(np.float32)

    coef = float(img_size[0] + img_size[1]) / (960 + 560)
    icon_radius = int(coef * 50)
    line_dist   = int(coef * 12)
    line_width  = int(coef *  6)
    font_size   = int(coef * 50)
    ah_size     = int(coef * 20) # arrow head size

    img = Image.new('LA', img_size, (0,0))
    draw = ImageDraw.Draw(img)

    temp = tr_prob.flatten().argsort()
    froms = np.floor(temp / n).astype(np.uint8)
    tos = (temp - n * froms).astype(np.uint8)
    del temp
    tr_prob_max = tr_prob[froms[-1], tos[-1]]

    # 90[deg] ccw rotation
    r90 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    # rotation for arrow head
    ang = 2 * np.pi / 10
    cos = np.cos(ang)
    sin = np.sin(ang)
    r_ah = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)

    # draw edge=transition probability
    for f, t in zip(froms, tos):
        # c = 255 - int(np.floor(255 * tr_prob[f,t] / tr_prob_max))
        strength = int(np.floor(255 * tr_prob[f,t] / tr_prob_max))

        if f == t: # Start and end are same => Draw semi-circle 
            box = [
                centers[f,0] + 0.5 * icon_radius, centers[f,1] - 0.7 * icon_radius,
                centers[f,0] + 2   * icon_radius, centers[f,1] + 0.7 * icon_radius
            ]
            box = [int(v) for v in box]
            draw.arc(
                box,
                0, 360,
                fill=(0, strength),width=line_width
            )
            ah_s = np.array([centers[f,0] + 1.92 * icon_radius, centers[f,1]]) # arrow head start
            ah_tan = np.array([-0.2, 0.98]) # arrow head tangent
        else: # Start and end are different => Draw straight line
            d = (centers[t] - centers[f])
            d = d / np.linalg.norm(d)
            ah_tan = d # arrow head tangent
            d = line_dist * r90 @ d
            s = centers[f] + d # arrow start
            e = centers[t] + d # arrow end
            box = [s[0], s[1], e[0], e[1]]
            box = [int(v) for v in box]
            ah_s = (s + e) / 2 # arrow head start
            draw.line(
                box,
                fill=(0,strength), width=line_width
            )

        # Draw arrow head
        ah_e = ah_s + ah_size * r_ah @ ah_tan
        bbox = [ah_s[0], ah_s[1], ah_e[0], ah_e[1]]
        bbox = [int(v) for v in bbox]
        draw.line(
            bbox,
            fill=(0,strength), width=line_width
        )
        ah_e = ah_s + ah_size * r_ah.T @ ah_tan
        bbox = [ah_s[0], ah_s[1], ah_e[0], ah_e[1]]
        bbox = [int(v) for v in bbox]
        draw.line(
            bbox,
            fill=(0,strength), width=line_width
        )

    # draw node=motion
    font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', size=font_size)
    for i in range(n):
        bbox = [
            centers[i,0] - icon_radius, centers[i,1] - icon_radius,
            centers[i,0] + icon_radius, centers[i,1] + icon_radius
        ]
        bbox = [int(v) for v in bbox]
        draw.ellipse(
            bbox,
            fill=(255,255), outline=(0,255), width=line_width
        )

        c = [centers[i,0], centers[i,1]]
        c = [int(v) for v in c]
        draw.text(
            c, '{}'.format(i + 1),
            font=font, anchor='mm', fill=(0, 255)
        )

    return img
