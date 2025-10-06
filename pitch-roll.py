import sys
import math
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# ----- CONFIG -----
PORT = 'COM5'  # <- change if needed
BAUD = 115200
WINDOW = 200                   # number of samples shown

ser = serial.Serial(PORT, BAUD, timeout=1)

pitch_buf = deque(maxlen=WINDOW)
roll_buf  = deque(maxlen=WINDOW)
x_idx     = deque(maxlen=WINDOW)

fig = plt.figure(figsize=(10,6))

# Top: time-series lines
ax1 = fig.add_subplot(2,1,1)
(line_pitch,) = ax1.plot([], [], label="Pitch (°)")
(line_roll,)  = ax1.plot([], [], label="Roll (°)")
ax1.set_xlim(0, WINDOW)
ax1.set_ylim(-90, 90)
ax1.set_xlabel("Samples")
ax1.set_ylabel("Angle (°)")
ax1.set_title("MPU6050 Pitch (Y) & Roll (X)")
ax1.legend(loc="upper right")

# Bottom: 3D pyramid
ax2 = fig.add_subplot(2,1,2, projection='3d')
ax2.set_xlim([-2,2])
ax2.set_ylim([-2,2])
ax2.set_zlim([-2,2])
ax2.set_box_aspect([1,1,1])
ax2.set_title("Pitch & Roll Driven Pyramid")

# Pyramid vertices (square base + apex)
h = 1.5  # height
r = 1.0  # half base size
vertices = np.array([
    [-r,-r,0],  # base corner 1
    [ r,-r,0],  # base corner 2
    [ r, r,0],  # base corner 3
    [-r, r,0],  # base corner 4
    [ 0, 0,h]   # apex
])

faces_idx = [
    [0,1,2,3],  # base
    [0,1,4],    # sides
    [1,2,4],
    [2,3,4],
    [3,0,4]
]

face_colors = ["lightgray", "cyan", "green", "orange", "red"]

pyramid = None

def rotation_matrix(pitch, roll):
    """Create rotation matrix from pitch & roll (in degrees)."""
    p = math.radians(pitch)
    r = math.radians(roll)

    Rx = np.array([
        [1,0,0],
        [0,math.cos(r),-math.sin(r)],
        [0,math.sin(r), math.cos(r)]
    ])

    Ry = np.array([
        [math.cos(p),0,math.sin(p)],
        [0,1,0],
        [-math.sin(p),0,math.cos(p)]
    ])

    return Ry @ Rx

def update_pyramid(pitch, roll):
    global pyramid
    rot = rotation_matrix(pitch, roll)
    new_vertices = np.dot(vertices, rot.T)
    new_faces = [[new_vertices[j] for j in f] for f in faces_idx]
    pyramid.set_verts(new_faces)

def parse_line(line):
    try:
        parts = line.strip().split(',')
        if len(parts) != 2:
            return None, None
        pitch = float(parts[0])
        roll = float(parts[1])
        return pitch, roll
    except:
        return None, None

def init():
    global pyramid
    faces = [[vertices[j] for j in f] for f in faces_idx]
    pyramid = Poly3DCollection(faces, facecolors=face_colors,
                               edgecolors="k", linewidths=1, alpha=0.8)
    ax2.add_collection3d(pyramid)
    update_pyramid(0,0)
    line_pitch.set_data([], [])
    line_roll.set_data([], [])
    return (line_pitch, line_roll, pyramid)

def update(frame):
    for _ in range(5):
        raw = ser.readline().decode(errors='ignore')
        if not raw:
            break
        pitch, roll = parse_line(raw)
        if pitch is None:
            continue
        pitch_buf.append(pitch)
        roll_buf.append(roll)
        x_idx.append(len(x_idx) + 1 if x_idx else 1)

    # Update time-series
    xs = list(range(len(x_idx)))
    line_pitch.set_data(xs, list(pitch_buf))
    line_roll.set_data(xs, list(roll_buf))
    ax1.set_xlim(max(0, len(xs)-WINDOW), max(WINDOW, len(xs)))

    # Update pyramid orientation
    if pitch_buf and roll_buf:
        update_pyramid(pitch_buf[-1], roll_buf[-1])

    return (line_pitch, line_roll, pyramid)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=30, blit=False)
plt.tight_layout()
plt.show()
