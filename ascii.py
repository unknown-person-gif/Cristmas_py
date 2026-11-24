# Man scrolling reels in a cave  (terminal / ASCII character grid)
# To run this pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

K = 3                                
W, H = 160 * K, 100 * K             
rng = np.random.default_rng(5)

yy, xx = np.mgrid[0:H, 0:W] / K

CHARS = "01010101<>/{}=+;$#"          
GW, GH = 5, 10                       
BX, BY = 2, 4                       
COLS, ROWS = W // BX, H // BY        


# helpers 
def circle(cx, cy, r):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

def rect(x0, y0, x1, y1):
    return (xx >= x0) & (xx < x1 + 1) & (yy >= y0) & (yy < y1 + 1)

def line(x0, y0, x1, y1, r=1.2):
    dx, dy = x1 - x0, y1 - y0
    u = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), 0, 1)
    return (xx - (x0 + u * dx)) ** 2 + (yy - (y0 + u * dy)) ** 2 <= r * r

def poly(points):
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


# ---------- face diamond ------------
def draw_gem(i, cx, cy, size, angle, base):
    
    R = size * 1.6
    x0, x1 = max(0, int((cx - R) * K)), min(W, int((cx + R) * K) + 2)
    y0, y1 = max(0, int((cy - R) * K)), min(H, int((cy + R) * K) + 2)
    gx, gy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
    c, s = np.cos(angle), np.sin(angle)
    u = ((gx - cx) * c + (gy - cy) * s) / size
    v = (-(gx - cx) * s + (gy - cy) * c) / size

    crown = (v >= -1) & (v < -0.4)
    wc = 0.6 + (v + 1) / 0.6 * 0.4                          
    wp = np.maximum(1 - (v + 0.4) / 1.4, 1e-3)              
    body = (crown & (abs(u) <= wc)) | ((v >= -0.4) & (v <= 1) & (abs(u) <= wp))
    if not body.any():
        return

    # facet shading 
    ratio = np.where(crown, u / wc, u / wp)
    crown_shade = np.select([ratio < -0.55, ratio < -0.2, ratio < 0.2, ratio < 0.55],
                            [0.95, 1.20, 1.38, 0.95], 0.72)
    pav_shade = np.select([ratio < -0.5, ratio < 0.0, ratio < 0.5], [1.12, 1.30, 0.85], 0.62)
    shade = np.where(crown, crown_shade, pav_shade)
    shade = shade * (1.12 - 0.22 * (v + 1) / 2)

    colour = np.array(base)[None, None, :] * shade[:, :, None]
    colour[(v > -0.46) & (v < -0.36) & body] = np.array((0.85, 0.97, 1.0))   # bright girdle band

    # thin edges
    thr = 0.5 / (K * size)
    edges = np.zeros_like(body)
    for b in (-0.55, -0.2, 0.2, 0.55):
        edges |= crown & (abs(u - b * wc) < thr)
    for b in (-0.5, 0.0, 0.5):
        edges |= (v >= -0.4) & (abs(u - b * wp) < thr)
    colour[edges & body] = np.array(base) * 0.55

    # outline
    inner = (((v >= -1) & (v < -0.4) & (abs(u) <= wc - 0.55 / size)) |
             ((v >= -0.4) & (v <= 1 - 0.7 / size) & (abs(u) <= wp - 0.55 / size)))
    colour[body & ~inner] = (0.05, 0.13, 0.30)

    # shiny sparks
    for gu, gv, gr in ((-0.28, -0.80, 0.13), (0.22, -0.62, 0.07), (-0.35, 0.05, 0.07)):
        colour[((u - gu) ** 2 + (v - gv) ** 2 < gr * gr) & body & inner] = (1, 1, 1)

    sub = bg[y0:y1, x0:x1]
    sub[body] = colour[body]
    gem_id[y0:y1, x0:x1][body] = i


rock = poly([(-2, 102), (-2, 44), (14, 36), (38, 34), (60, 38), (68, 52), (70, 72), (66, 88), (74, 96), (74, 102)])
inside = poly([(-2, 52), (40, 48), (54, 52), (56, 66), (56, 92), (-2, 92)])

gem_centers = []
for gx in np.arange(0, 160, 9.5):
    for gy in np.arange(0, 100, 9.5):
        cx, cy = gx + rng.uniform(-3.5, 3.5), gy + rng.uniform(-3.5, 3.5)
        if cx < 76 and cy > 32:                          # To keep the cave area empty
            continue
        size, angle = rng.uniform(4.2, 6.2), rng.uniform(-0.5, 0.5)
        draw_gem(len(gem_centers), cx, cy, size, angle, GEM_COLOURS[rng.integers(len(GEM_COLOURS))])
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

# man 
bg[line(40, 76, 33, 91, 1.0)] = (0.35, 0.2, 0.08)                            # axe
bg[line(35, 72, 46, 78, 1.5)] = (0.3, 0.3, 0.34)
bg[rect(19, 80, 23, 91) | rect(25, 80, 29, 91)] = (0.08, 0.08, 0.1)          # legs
bg[rect(18, 91, 23, 92) | rect(25, 91, 30, 92)] = (0, 0, 0)                  # shoes
bg[poly([(17, 64), (31, 64), (32, 80), (16, 80)])] = HOODIE               # hoodie
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


# ---------- one frame of the scene ----------
def draw_scene(t):
    img = bg.copy()

    # diamonds shimmer
    shade = 0.92 + 0.12 * np.sin(2 * t + gem_phase)
    img *= np.where(gem_id >= 0, shade[gem_id.clip(0)], 1.0)[:, :, None]

    # blue, red light
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

    # sparkles on diamonds (4-point stars)
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


# ---------- ASCII / terminal ----------
def build_atlas():
    
    atlas = []
    for ch in CHARS:
        f = Figure(figsize=(GW / 100, GH / 100), dpi=100, facecolor="black")
        canvas = FigureCanvasAgg(f)
        f.text(0.5, 0.5, ch, color="white", family="monospace", fontsize=6,
               fontweight="bold", ha="center", va="center")
        canvas.draw()
        atlas.append(np.asarray(canvas.buffer_rgba())[:GH, :GW, 0] / 255.0)
    return np.array(atlas)                              

ATLAS = build_atlas()

CHAR_MAP = np.random.default_rng(11).integers(0, len(CHARS), (ROWS, COLS))
GLYPHS = ATLAS[CHAR_MAP][..., None]                      


def ascii_render(img):
   
    cells = img.reshape(ROWS, BY, COLS, BX, 3).mean(axis=(1, 3))         
    cells = cells[:, :, None, None, :]
    fg = np.clip(cells * 1.1 + 0.04, 0, 1)                                
    back = cells * 0.62                                                   
    out = back * (1 - GLYPHS) + fg * GLYPHS
    return out.transpose(0, 2, 1, 3, 4).reshape(ROWS * GH, COLS * GW, 3)


# ---------- window ----------
fig = plt.figure(figsize=(COLS * GW / 100, ROWS * GH / 100 / 0.93), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0.07, 1, 0.93])
ax.axis("off")
im = ax.imshow(ascii_render(draw_scene(0)), interpolation="nearest")
fig.text(0.5, 0.03, "> He is scrolling.  The diamonds are right outside._", color="#33ff66",
         family="monospace", fontsize=14, fontweight="bold", ha="center", va="center")


def update(frame):
    im.set_data(ascii_render(draw_scene(frame / 20)))
    return (im,)


ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()
