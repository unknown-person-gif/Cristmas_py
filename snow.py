import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

W, H = 170, 100
GROUND = 78
N_FLAKES = 350
rng = np.random.default_rng(7)

flakes = pd.DataFrame({
    "x": rng.uniform(0, W, N_FLAKES),
    "y": rng.uniform(0, H, N_FLAKES),
    "speed": rng.uniform(0.4, 1.4, N_FLAKES),
    "drift": rng.uniform(-0.3, 0.3, N_FLAKES),
    "phase": rng.uniform(0, 6.28, N_FLAKES),
})
stars = pd.DataFrame({"x": rng.integers(0, W, 40), "y": rng.integers(0, 45, 40),
                      "phase": rng.uniform(0, 6.28, 40)})

yy, xx = np.mgrid[0:H, 0:W]


def circle(cx, cy, r):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def rect(x0, y0, x1, y1):
    return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)


def seg(x0, y0, x1, y1, r=1.4):
    
    dx, dy = x1 - x0, y1 - y0
    L = dx * dx + dy * dy + 1e-9
    u = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L, 0, 1)
    return (xx - (x0 + u * dx)) ** 2 + (yy - (y0 + u * dy)) ** 2 <= r * r


SKIN = (0.96, 0.76, 0.62)
RED = (0.9, 0.12, 0.12)
WHITE = (0.95, 0.95, 0.98)


def santa_hat(img, cx, top, side=1):
    img[rect(cx - 5, top, cx + 5, top + 2)] = WHITE
    tri = (yy >= top - 6) & (yy < top) & (np.abs(xx - cx - side * (top - yy) * 0.3) <= (yy - (top - 6)) * 0.8)
    img[tri] = RED
    img[circle(cx + side * 3, top - 6, 1.6)] = WHITE


def draw_girl(img, t, gx, hand):
    s = t * 4
    bob = abs(np.sin(s)) * 2
    sway = np.sin(s) * 1.2
    step = 3 * np.sin(s)
    # legs
    img[seg(gx - 2, 72 - bob, gx - 2 + step, GROUND - max(0, np.sin(s)) * 2, 1.3)] = SKIN
    img[seg(gx + 2, 72 - bob, gx + 2 - step, GROUND - max(0, -np.sin(s)) * 2, 1.3)] = SKIN
    # dress
    dress = (yy >= 56 - bob) & (yy <= 72 - bob) & (np.abs(xx - gx - sway) <= 3 + (yy - 56 + bob) * 0.45)
    img[dress] = (0.85, 0.1, 0.35)
    img[rect(gx - 7 + sway, 71 - bob, gx + 7 + sway, 72 - bob)] = WHITE
    # arms: one holds partner's hand, one waves
    img[seg(gx + 3, 58 - bob, hand[0], hand[1], 1.2)] = SKIN
    img[seg(gx - 3, 58 - bob, gx - 10, 50 - bob + 5 * np.sin(s + 1), 1.2)] = SKIN
    # head, hair, hat
    img[circle(gx, 50 - bob, 5)] = SKIN
    img[rect(gx - 5, 44 - bob, gx + 5, 49 - bob) & circle(gx, 50 - bob, 5.4)] = (0.45, 0.25, 0.1)
    img[rect(gx - 6, 46 - bob, gx - 4, 58 - bob)] = (0.45, 0.25, 0.1)
    img[rect(gx + 4, 46 - bob, gx + 6, 58 - bob)] = (0.45, 0.25, 0.1)
    santa_hat(img, gx, 45 - bob, -1)
    img[circle(gx - 2, 50 - bob, 0.7)] = (0, 0, 0)
    img[circle(gx + 2, 50 - bob, 0.7)] = (0, 0, 0)


def draw_boy(img, t, bx, hand):
    s = t * 4 + 3.14
    bob = abs(np.sin(s)) * 2
    step = 3 * np.sin(s)
    img[seg(bx - 2, 68 - bob, bx - 2 + step, GROUND - max(0, np.sin(s)) * 2, 1.5)] = (0.15, 0.2, 0.5)
    img[seg(bx + 2, 68 - bob, bx + 2 - step, GROUND - max(0, -np.sin(s)) * 2, 1.5)] = (0.15, 0.2, 0.5)
    img[rect(bx - 4, 56 - bob, bx + 4, 69 - bob)] = (0.1, 0.55, 0.25)     # torso
    img[rect(bx - 4, 62 - bob, bx + 4, 63 - bob)] = (0.9, 0.8, 0.2)       # belt
    img[seg(bx - 4, 58 - bob, hand[0], hand[1], 1.3)] = SKIN
    img[seg(bx + 4, 58 - bob, bx + 11, 50 - bob + 5 * np.sin(s + 1), 1.3)] = SKIN
    img[circle(bx, 50 - bob, 5)] = SKIN
    img[rect(bx - 5, 45 - bob, bx + 5, 48 - bob) & circle(bx, 50 - bob, 5.4)] = (0.2, 0.12, 0.05)
    santa_hat(img, bx, 45 - bob, 1)
    img[circle(bx - 2, 50 - bob, 0.7)] = (0, 0, 0)
    img[circle(bx + 2, 50 - bob, 0.7)] = (0, 0, 0)


def draw_tree(img, t, cx):
    for i, (top, half) in enumerate([(40, 9), (52, 13), (64, 17)]):
        tri = (yy >= top) & (yy <= top + 16) & (np.abs(xx - cx) <= (yy - top) * half / 16)
        img[tri] = (0.05, 0.45, 0.15)
    img[rect(cx - 2, 78 - 4, cx + 2, 78)] = (0.4, 0.22, 0.08)
    colors = [(1, .2, .2), (1, .9, .2), (.3, .6, 1), (1, .5, .9)]
    for k, (lx, ly) in enumerate([(cx, 44), (cx - 4, 54), (cx + 5, 58), (cx - 9, 68), (cx + 2, 66), (cx + 10, 70)]):
        on = 0.5 + 0.5 * np.sin(t * 5 + k * 1.7)
        img[circle(lx, ly, 1.4)] = np.array(colors[k % 4]) * (0.4 + 0.6 * on)
    # star
    img[circle(cx, 37, 2.2)] = np.array([1, 0.9, 0.3]) * (0.7 + 0.3 * np.sin(t * 6))


def build_scene(t):
    img = np.zeros((H, W, 3))
    img[:] = (0.02, 0.03, 0.09)
    for _, st in stars.iterrows():
        img[int(st.y), int(st.x)] = np.array([1, 1, 0.9]) * (0.4 + 0.6 * abs(np.sin(t * 2 + st.phase)))
    img[yy >= GROUND] = (0.85, 0.9, 0.95)

    # lamp post
    img[rect(13, 30, 15, GROUND)] = (0.15, 0.15, 0.18)
    glow = 0.85 + 0.15 * np.sin(t * 3)
    lamp = rect(9, 20, 19, 31)
    dist = np.sqrt((xx - 14) ** 2 + (yy - 25) ** 2)
    halo = (dist <= 22) & ~lamp
    img[halo] += (np.clip(1 - dist[halo] / 22, 0, 1) ** 2)[:, None] * (0.5, 0.35, 0.1)
    img[lamp] = (glow, 0.75 * glow, 0.25 * glow)

    # dancers
    gx, bx = 58, 82
    bob = abs(np.sin(t * 4)) * 2
    hand = (( gx + bx) / 2, 60 - bob + 1.5 * np.sin(t * 4))
    draw_girl(img, t, gx, hand)
    draw_boy(img, t, bx, hand)

    # music notes floating up
    for k in range(4):
        p = ((t * 0.35 + k * 0.25) % 1)
        nx = 52 + k * 16 + 4 * np.sin(t * 2 + k)
        ny = 42 - p * 36
        a = 1 - p
        img[circle(nx, ny, 1.6)] = np.array([1, 0.9, 0.3]) * a
        img[rect(nx + 1, ny - 6, nx + 1.6, ny)] = np.array([1, 0.9, 0.3]) * a

    # tree + snowman
    draw_tree(img, t, 150)
    sw = np.sin(t * 1.5) * 0.8
    sx = 118
    img[circle(sx + sw, 72, 9)] = (0.35, 0.65, 0.9)
    img[circle(sx + sw * 1.5, 58, 7)] = (0.4, 0.7, 0.95)
    img[rect(sx - 7 + sw, 63, sx + 7 + sw, 66)] = (0.9, 0.55, 0.1)
    img[rect(sx - 6 + sw, 48, sx + 6 + sw, 51)] = RED
    img[rect(sx - 4 + sw, 42, sx + 4 + sw, 49)] = RED
    img[circle(sx - 2.5 + sw * 1.5, 56, 0.9)] = (0, 0, 0)
    img[circle(sx + 2.5 + sw * 1.5, 56, 0.9)] = (0, 0, 0)
    img[circle(sx + sw * 1.5, 59, 1.2)] = (1, 0.5, 0)

    # snow
    fx = (flakes["x"] + np.sin(t + flakes["phase"]) * 2).astype(int) % W
    fy = flakes["y"].astype(int) % H
    img[fy, fx] = (1, 1, 1)
    return np.clip(img, 0, 1)


def dot_matrix(img, scale=4):
    big = np.kron(img, np.ones((scale, scale, 1)))
    c = (scale - 1) / 2
    ty, tx = np.mgrid[0:scale, 0:scale]
    tile = ((tx - c) ** 2 + (ty - c) ** 2 <= (scale / 2) ** 2).astype(float)
    return big * np.tile(tile, (H, W))[:, :, None]


fig, ax = plt.subplots(figsize=(9, 5.4))
fig.patch.set_facecolor("black")
ax.axis("off")
im = ax.imshow(dot_matrix(build_scene(0)), interpolation="nearest")
ax.set_title("Merry Christmas!", color="white", fontsize=14, pad=8)
plt.tight_layout()


def update(frame):
    t = frame / 20
    flakes["y"] = (flakes["y"] + flakes["speed"]) % H
    flakes["x"] = (flakes["x"] + flakes["drift"]) % W
    im.set_data(dot_matrix(build_scene(t)))
    return (im,)


anim = FuncAnimation(fig, update, frames=160, interval=50, blit=True)

if "--save" in sys.argv:
    anim.save("snow_globe.gif", writer=PillowWriter(fps=20))
    print("Saved snow_globe.gif")
else:
    plt.show()