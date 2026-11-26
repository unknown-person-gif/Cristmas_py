import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Screen settings var declarations :)
W, H = 170, 100
GROUND = 78
TOTAL_SNOW = 350

rng = np.random.default_rng(7)

# Random positions and movement for snowflakes
snow_x = rng.uniform(0, W, TOTAL_SNOW)
snow_y = rng.uniform(0, H, TOTAL_SNOW)
snow_speed = rng.uniform(0.4, 1.4, TOTAL_SNOW)
snow_drift = rng.uniform(-0.3, 0.3, TOTAL_SNOW)
snow_phase = rng.uniform(0, 6.28, TOTAL_SNOW)

# Background stars
star_x = rng.integers(0, W, 40)
star_y = rng.integers(0, 45, 40)
star_phase = rng.uniform(0, 6.28, 40)

yy, xx = np.mgrid[0:H, 0:W]

SKIN = (0.96, 0.76, 0.62)
RED = (0.9, 0.12, 0.12)
WHITE = (0.95, 0.95, 0.98)


# Draw a circle
def circle(x, y, r):
    return (xx - x)**2 + (yy - y)**2 <= r**2


# Draw a rectangle
def rectangle(x1, y1, x2, y2):
    return (
        (xx >= x1) & (xx <= x2) &
        (yy >= y1) & (yy <= y2)
    )


# Draw a line with thickness
def line(x1, y1, x2, y2, thickness=1.4):
    dx = x2 - x1
    dy = y2 - y1

    length = dx**2 + dy**2 + 1e-9

    pos = np.clip(
        ((xx - x1) * dx + (yy - y1) * dy) / length,
        0, 1
    )

    return (
        (xx - (x1 + pos * dx))**2 +
        (yy - (y1 + pos * dy))**2 <= thickness**2
    )


# Santa hat
def draw_hat(img, x, y, direction=1):
    img[rectangle(x-5, y, x+5, y+2)] = WHITE

    hat = (
        (yy >= y-6) & (yy < y) &
        (np.abs(xx-x-direction*(y-yy)*0.3)
         <= (yy-(y-6))*0.8)
    )

    img[hat] = RED
    img[circle(x + direction*3, y-6, 1.6)] = WHITE


# Draw the girl
def draw_girl(img, t, x, hand):
    s = t * 4

    bob = abs(np.sin(s)) * 2
    sway = np.sin(s) * 1.2
    step = 3 * np.sin(s)

    # Legs
    img[line(x-2, 72-bob, x-2+step, GROUND, 1.3)] = SKIN
    img[line(x+2, 72-bob, x+2-step, GROUND, 1.3)] = SKIN

    # Dress
    dress = (
        (yy >= 56-bob) &
        (yy <= 72-bob) &
        (np.abs(xx-x-sway) <= 3+(yy-56+bob)*0.45)
    )

    img[dress] = (0.85, 0.1, 0.35)
    img[rectangle(x-7+sway, 71-bob, x+7+sway, 72-bob)] = WHITE

    # Arms
    img[line(x+3, 58-bob, hand[0], hand[1], 1.2)] = SKIN
    img[line(x-3, 58-bob, x-10, 50-bob+5*np.sin(s+1), 1.2)] = SKIN

    # Face and hair
    img[circle(x, 50-bob, 5)] = SKIN

    hair = rectangle(x-5, 44-bob, x+5, 49-bob) & circle(x, 50-bob, 5.4)
    img[hair] = (0.45, 0.25, 0.1)

    img[rectangle(x-6, 46-bob, x-4, 58-bob)] = (0.45, 0.25, 0.1)
    img[rectangle(x+4, 46-bob, x+6, 58-bob)] = (0.45, 0.25, 0.1)

    draw_hat(img, x, 45-bob, -1)

    # Eyes
    img[circle(x-2, 50-bob, 0.7)] = (0, 0, 0)
    img[circle(x+2, 50-bob, 0.7)] = (0, 0, 0)


# Draw the boy
def draw_boy(img, t, x, hand):
    s = t * 4 + np.pi

    bob = abs(np.sin(s)) * 2
    step = 3 * np.sin(s)

    # Legs
    img[line(x-2, 68-bob, x-2+step, GROUND, 1.5)] = (0.15, 0.2, 0.5)
    img[line(x+2, 68-bob, x+2-step, GROUND, 1.5)] = (0.15, 0.2, 0.5)

    # Shirt and belt
    img[rectangle(x-4, 56-bob, x+4, 69-bob)] = (0.1, 0.55, 0.25)
    img[rectangle(x-4, 62-bob, x+4, 63-bob)] = (0.9, 0.8, 0.2)

    # Arms
    img[line(x-4, 58-bob, hand[0], hand[1], 1.3)] = SKIN
    img[line(x+4, 58-bob, x+11, 50-bob+5*np.sin(s+1), 1.3)] = SKIN

    # Face and hair
    img[circle(x, 50-bob, 5)] = SKIN

    hair = rectangle(x-5, 45-bob, x+5, 48-bob) & circle(x, 50-bob, 5.4)
    img[hair] = (0.2, 0.12, 0.05)

    draw_hat(img, x, 45-bob, 1)

    img[circle(x-2, 50-bob, 0.7)] = (0, 0, 0)
    img[circle(x+2, 50-bob, 0.7)] = (0, 0, 0)


# Christmas tree with blinking lights
def draw_tree(img, t, x):
    for top, width in [(40, 9), (52, 13), (64, 17)]:
        tree_part = (
            (yy >= top) &
            (yy <= top+16) &
            (np.abs(xx-x) <= (yy-top)*width/16)
        )

        img[tree_part] = (0.05, 0.45, 0.15)

    # Tree trunk
    img[rectangle(x-2, 74, x+2, 78)] = (0.4, 0.22, 0.08)

    # Decorative lights
    colors = [
        (1, 0.2, 0.2),
        (1, 0.9, 0.2),
        (0.3, 0.6, 1),
        (1, 0.5, 0.9)
    ]

    lights = [
        (x, 44), (x-4, 54), (x+5, 58),
        (x-9, 68), (x+2, 66), (x+10, 70)
    ]

    for i, (lx, ly) in enumerate(lights):
        brightness = 0.4 + 0.6 * (0.5 + 0.5*np.sin(t*5+i*1.7))
        img[circle(lx, ly, 1.4)] = np.array(colors[i % 4]) * brightness

    # Star on top
    img[circle(x, 37, 2.2)] = (1, 0.9, 0.3)


# Create the complete Christmas scene
def build_scene(t):
    img = np.zeros((H, W, 3))
    img[:] = (0.02, 0.03, 0.09)

    # Stars in the sky
    for i in range(len(star_x)):
        brightness = 0.4 + 0.6*abs(np.sin(t*2+star_phase[i]))
        img[star_y[i], star_x[i]] = (brightness, brightness, brightness*0.9)

    # Snow-covered ground
    img[yy >= GROUND] = (0.85, 0.9, 0.95)

    # Street lamp
    img[rectangle(13, 30, 15, GROUND)] = (0.15, 0.15, 0.18)

    glow = 0.85 + 0.15*np.sin(t*3)

    lamp = rectangle(9, 20, 19, 31)
    distance = np.sqrt((xx-14)**2+(yy-25)**2)

    halo = (distance <= 22) & ~lamp

    img[halo] += (
        np.clip(1-distance[halo]/22, 0, 1)**2
    )[:, None] * (0.5, 0.35, 0.1)

    img[lamp] = (glow, 0.75*glow, 0.25*glow)

    # Dancing couple
    girl_x = 58
    boy_x = 82

    bob = abs(np.sin(t*4))*2
    hand = (
        (girl_x+boy_x)/2,
        60-bob+1.5*np.sin(t*4)
    )

    draw_girl(img, t, girl_x, hand)
    draw_boy(img, t, boy_x, hand)

    # Floating music notes
    for i in range(4):
        p = (t*0.35+i*0.25) % 1

        nx = 52+i*16+4*np.sin(t*2+i)
        ny = 42-p*36

        brightness = 1-p
        color = np.array([1, 0.9, 0.3])*brightness

        img[circle(nx, ny, 1.6)] = color
        img[rectangle(nx+1, ny-6, nx+1.6, ny)] = color

    # Christmas tree
    draw_tree(img, t, 150)

    # Snowman
    move = np.sin(t*1.5)*0.8
    x = 118

    img[circle(x+move, 72, 9)] = (0.35, 0.65, 0.9)
    img[circle(x+move*1.5, 58, 7)] = (0.4, 0.7, 0.95)

    # Scarf and hat
    img[rectangle(x-7+move, 63, x+7+move, 66)] = (0.9, 0.55, 0.1)
    img[rectangle(x-6+move, 48, x+6+move, 51)] = RED
    img[rectangle(x-4+move, 42, x+4+move, 49)] = RED

    # Eyes and nose
    img[circle(x-2.5+move*1.5, 56, 0.9)] = (0, 0, 0)
    img[circle(x+2.5+move*1.5, 56, 0.9)] = (0, 0, 0)
    img[circle(x+move*1.5, 59, 1.2)] = (1, 0.5, 0)

    # Falling snow
    fx = (snow_x + np.sin(t+snow_phase)*2).astype(int) % W
    fy = snow_y.astype(int) % H

    img[fy, fx] = WHITE

    return np.clip(img, 0, 1)


# Convert the scene into a dotted LED-style display
def dot_matrix(img, scale=4):
    big = np.kron(img, np.ones((scale, scale, 1)))

    center = (scale-1)/2
    ty, tx = np.mgrid[0:scale, 0:scale]

    dots = (
        (tx-center)**2+(ty-center)**2 <= (scale/2)**2
    ).astype(float)

    return big*np.tile(dots, (H, W))[:, :, None]


# Display setup
fig, ax = plt.subplots(figsize=(9, 5.4))

fig.patch.set_facecolor("black")
ax.axis("off")

image = ax.imshow(dot_matrix(build_scene(0)), interpolation="nearest")
ax.set_title("Merry Christmas!", color="white", fontsize=14)

plt.tight_layout()


# Update animation frame
def update(frame):
    t = frame/20

    snow_y[:] = (snow_y+snow_speed) % H
    snow_x[:] = (snow_x+snow_drift) % W

    image.set_data(dot_matrix(build_scene(t)))

    return (image,)


animation = FuncAnimation(                                                       #HELLO LOOKING SOMETHING INTRESTING HERE ?
    fig,
    update,
    frames=160,
    interval=50,
    blit=True
)


# For Saving it as GIF or show the animation
if "--save" in sys.argv:
    animation.save(
        "snow_globe.gif",
        writer=PillowWriter(fps=20)
    )
    print("Christmas animation saved successfully!")

else:
    plt.show()
