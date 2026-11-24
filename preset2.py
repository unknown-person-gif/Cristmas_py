# Man scrolling reels in a cave 
# for pip install numpy matplotlib :)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation                                          # WHAT ARE YOU LOOKING FOR :XD

K = 3                                
SCALE = 6 // K                       
W, H = 160 * K, 100 * K              
rng = np.random.default_rng(5)

yy, xx = np.mgrid[0:H, 0:W] / K                         

def circle(cx, cy, r):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

def rect(x0, y0, x1, y1):
    return (xx >= x0) & (xx < x1 + 1) & (yy >= y0) & (yy < y1 + 1)

def line(x0, y0, x1, y1, r=1.2):
    dx, dy = x1 - x0, y1 - y0
    u = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), 0, 1)
    return (xx - (x0 + u * dx)) ** 2 + (yy - (y0 + u * dy)) ** 2 <= r * r

def polygon(points):
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return Path(points).contains_points(pts).reshape(H, W)

def outline(mask, thick=1.0):
    n = max(1, int(round(thick * K)))
    inner = mask.copy()
    for axis in (0, 1):
        for s in range(1, n + 1):
            inner &= np.roll(mask, s, axis) & np.roll(mask, -s, axis)
    edge = mask & ~inner
    edge[:2 * K], edge[-2 * K:], edge[:, :2 * K], edge[:, -2 * K:] = False, False, False, False
    return edge

def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

# ---------- colours ----------
SKY = (0.70, 0.92, 0.96)
BROWN, DARK_BROWN = (0.62, 0.36, 0.16), (0.30, 0.15, 0.06)
CAVE_GREY = (0.72, 0.72, 0.74)
HOODIE, SKIN = (0.47, 0.47, 0.52), np.array((0.93, 0.80, 0.70))
BLUE, RED = np.array((0.15, 0.45, 1.0)), np.array((1.0, 0.15, 0.25))
GEM_COLOURS = [(0.35, 0.72, 0.96), (0.28, 0.60, 0.92), (0.50, 0.86, 1.00),
               (0.25, 0.52, 0.88), (0.42, 0.78, 0.98)]
REELS = np.array([(0.95, 0.35, 0.5), (0.3, 0.7, 0.95), (1.0, 0.75, 0.2), (0.5, 0.4, 0.95)])

bg = np.zeros((H, W, 3))
bg[:] = SKY
gem_id = np.full((H, W), -1)

#---for diamonds ---
SPRITES = [
    [  # big diamond
        "..oooooo..",
        ".olwlllbo.",
        "olwllllbbo",
        "oooooooooo",
        ".olllbbdo.",
        "..olbbdo..",
        "...obdo...",
        "....oo....",
    ],
    [  # small diamond
        ".ooooo.",
        "olwllbo",
        "ooooooo",
        ".olbdo.",
        "..obo..",
        "...o...",
    ],
]
OUTLINE = np.array((0.05, 0.13, 0.30))


def draw_gem(i, cx, cy, sprite, base):
    base = np.array(base)
    pal = {"o": OUTLINE, "w": np.array((1.0, 1.0, 1.0)), "l": np.clip(base * 1.25, 0, 1),
           "b": base, "d": base * 0.68}
    h, w = len(sprite), len(sprite[0])
    ox, oy = int(round(cx - w / 2)) * K, int(round(cy - h / 2)) * K     
    for r, row in enumerate(sprite):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            ya, xa = oy + r * K, ox + c * K
            ya0, xa0, ya1, xa1 = max(ya, 0), max(xa, 0), min(ya + K, H), min(xa + K, W)
            if ya0 >= ya1 or xa0 >= xa1:
                continue
            bg[ya0:ya1, xa0:xa1] = pal[ch]
            gem_id[ya0:ya1, xa0:xa1] = i

rock = polygon([(-2, 102), (-2, 44), (14, 36), (38, 34), (60, 38), (68, 52), (70, 72), (66, 88), (74, 96), (74, 102)])
inside = polygon([(-2, 52), (40, 48), (54, 52), (56, 66), (56, 92), (-2, 92)])

gem_centers = []
for gx in np.arange(4, 160, 12):
    for gy in np.arange(4, 100, 12):
        if rng.random() < 0.15:                             
            continue
        cx, cy = gx + rng.uniform(-2, 2), gy + rng.uniform(-2, 2)
        if cx < 80 and cy > 28:                              
            continue
        sprite = SPRITES[0] if rng.random() < 0.65 else SPRITES[1]
        draw_gem(len(gem_centers), cx, cy, sprite, GEM_COLOURS[rng.integers(len(GEM_COLOURS))])
        gem_centers.append((cx, cy))
gem_phase = rng.uniform(0, 6.28, len(gem_centers))
spark_gems = rng.choice(len(gem_centers), 22, replace=False)
spark_speed = rng.uniform(1.5, 4, 22)

# rock + cave
bg[rock] = BROWN
bg[outline(rock, 1.2)] = DARK_BROWN
bg[inside] = CAVE_GREY
bg[outline(inside, 1.0)] = DARK_BROWN
for px, py in [(46, 90), (49, 89), (43, 91)]:
    bg[circle(px, py, 1.5)] = (0.55, 0.32, 0.14)

# Completee man 
bg[line(40, 76, 33, 91, 1.0)] = (0.35, 0.2, 0.08)                            # axe
bg[line(35, 72, 46, 78, 1.5)] = (0.3, 0.3, 0.34)
bg[rect(19, 80, 23, 91) | rect(25, 80, 29, 91)] = (0.08, 0.08, 0.1)          # legs
bg[rect(18, 91, 23, 92) | rect(25, 91, 30, 92)] = (0, 0, 0)                  # shoes
bg[polygon([(17, 64), (31, 64), (32, 80), (16, 80)])] = HOODIE               # hoodie
bg[circle(24, 56, 8)] = (0.40, 0.40, 0.45)                                   # hood
bg[line(17, 66, 18, 74, 1.6) | line(18, 74, 21, 74, 1.6)] = HOODIE           # arms
bg[line(31, 66, 30, 74, 1.6) | line(30, 74, 27, 74, 1.6)] = HOODIE
bg[rect(21, 68, 27, 80)] = (0.05, 0.05, 0.07)                                # phone
bg[circle(21, 74, 1.5) | circle(27, 74, 1.5)] = SKIN                         # hands

head = circle(24, 57, 6)
screen = rect(22, 69, 26, 79)
screen_x, screen_y = xx - 22, yy - 69
HEART = np.kron(np.array([[0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool),
                np.ones((K, K), bool))


# ---------- one frame ----------
def draw_scene(t):
    img = bg.copy()

    # diamonds 
    shade = 0.92 + 0.12 * np.sin(2 * t + gem_phase)
    img *= np.where(gem_id >= 0, shade[gem_id.clip(0)], 1.0)[:, :, None]

    # blue, red light, flickering
    mix = 0.5 + 0.5 * np.sin(2 * np.pi * t / 1.5)
    light = BLUE * (1 - mix) + RED * mix
    flicker = 0.7 + 0.3 * np.sin(2 * np.pi * 4 * t) ** 2
    d = np.sqrt((xx - 24) ** 2 + (yy - 62) ** 2)
    s = (np.clip(1 - d / 40, 0, 1) ** 1.5 * 0.55 * flicker * inside)[:, :, None]
    img = img * (1 - s) + light * s

    # reels scrolling on the phone
    k = int(t // 1.5)
    offset = (k + smooth((t - 1.5 * k - 1.0) / 0.4)) * 11
    p = screen_y + offset
    card, local = np.floor(p / 11).astype(int), p % 11
    colour = REELS[card % len(REELS)]
    white = ((local >= 8) & (local < 10) & (screen_x < 3)) | ((local >= 2) & (local < 3) & (screen_x < 1))
    colour[white] = (1, 1, 1)
    img[screen] = colour[screen]

    # face lit by the screen
    img[head] = SKIN * (1 - 0.5 * flicker) + light * 0.5 * flicker
    blink = abs(t % 6 - 1.3) < 0.1 or abs(t % 6 - 4.1) < 0.1
    if not blink:
        img[rect(21, 57, 22, 58) | rect(26, 57, 27, 58)] = (1, 1, 1)
        img[rect(22, 58, 22, 58) | rect(27, 58, 27, 58)] = (0, 0, 0)
    img[rect(23, 62, 25, 62)] = (0.25, 0.08, 0.08)

    # hearts floating up
    for i in range(3):
        q = (t / 2 + i / 3) % 1
        hx, hy = int((34 + 4 * np.sin(6.28 * q + i)) * K), int((70 - q * 28) * K)
        a = np.sin(np.pi * q)
        area = img[hy:hy + HEART.shape[0], hx:hx + HEART.shape[1]]
        area[HEART] = area[HEART] * (1 - a) + np.array((1, 0.2, 0.35)) * a

    # sparkles on diamonds 
    for g, sp in zip(spark_gems, spark_speed):
        a = max(0, np.sin(sp * t)) ** 6
        if a < 0.03:
            continue
        cx, cy = gem_centers[g]
        x0, x1 = max(0, int((cx - 6) * K)), min(W, int((cx + 6) * K))
        y0, y1 = max(0, int((cy - 6) * K)), min(H, int((cy + 6) * K))
        dx, dy = np.abs(xx[y0:y1, x0:x1] - cx), np.abs(yy[y0:y1, x0:x1] - cy)
        arm = 1.5 + 4 * a
        star = ((dx < 0.3) & (dy < arm * (1 - dy / 9))) | ((dy < 0.3) & (dx < arm * (1 - dx / 9))) | (dx ** 2 + dy ** 2 < 0.8)
        area = img[y0:y1, x0:x1]
        area[star] = area[star] * (1 - a) + a

    return np.clip(img, 0, 1)


GRID_DARKNESS = 0.35                

def dot_matrix(img):
    big = np.kron(img, np.ones((SCALE, SCALE, 1)))
    tile = np.ones((SCALE, SCALE))
    tile[0, :] = tile[:, 0] = 1 - GRID_DARKNESS          # For one-pixel gapping
    return big * np.tile(tile, (H, W))[:, :, None]


# ---------- window ----------
fig = plt.figure(figsize=(W * SCALE / 100, H * SCALE / 100 / 0.93), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0.07, 1, 0.93])
ax.axis("off")
im = ax.imshow(dot_matrix(draw_scene(0)), interpolation="nearest")
fig.text(0.5, 0.03, "i need to get in .", color="white",
         fontsize=14, fontweight="bold", ha="center", va="center")                                       # hello elliot


def update(frame):
    im.set_data(dot_matrix(draw_scene(frame / 20)))
    return (im,)


ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()
