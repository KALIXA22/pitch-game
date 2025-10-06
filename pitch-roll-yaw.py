import sys
import math
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# ----- CONFIG -----
PORT = 'COM5'
BAUD = 115200
WINDOW = 200

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"✅ Serial opened on {PORT} at {BAUD}")
except Exception as e:
    print(f"❌ Could not open serial: {e}")
    sys.exit()

# Buffers
pitch_buf = deque(maxlen=WINDOW)
roll_buf = deque(maxlen=WINDOW)
yaw_buf = deque(maxlen=WINDOW)
x_idx = deque(maxlen=WINDOW)

# ----- FIGURE -----
fig = plt.figure(figsize=(10,7))

# Top: time-series
ax1 = fig.add_subplot(2,1,1)
line_pitch, = ax1.plot([], [], label="Pitch (°)")
line_roll, = ax1.plot([], [], label="Roll (°)")
line_yaw, = ax1.plot([], [], label="Yaw (°)")
ax1.set_xlim(0, WINDOW)
ax1.set_ylim(-180,180)
ax1.set_xlabel("Samples")
ax1.set_ylabel("Angle (°)")
ax1.set_title("MPU6050 Orientation")
ax1.legend(loc="upper right")

# Bottom: 3D pyramid
ax2 = fig.add_subplot(2,1,2, projection='3d')
ax2.set_xlim([-2,2])
ax2.set_ylim([-2,2])
ax2.set_zlim([0,2])
ax2.set_box_aspect([1,1,1])
ax2.set_title("Pyramid Orientation (Front = Red, Back = Blue)")

# ----- PYRAMID MODEL -----
h = 1.5   # height
r = 1.0   # half base size
vertices = np.array([
    [-r,-r,0],  # base corner 1
    [ r,-r,0],  # base corner 2
    [ r, r,0],  # base corner 3
    [-r, r,0],  # base corner 4
    [ 0, 0,h]   # apex
])

# Faces: base + four sides
faces_idx = [
    [0,1,2,3],  # base
    [0,1,4],    # side front (red)
    [1,2,4],    # side right
    [2,3,4],    # side back (blue)
    [3,0,4]     # side left
]

face_colors = ["lightgray","red","cyan","blue","cyan"]

pyramid = None

# Rotation matrix (pitch, roll, yaw)
def rotation_matrix(pitch, roll, yaw):
    p = math.radians(pitch)
    r = math.radians(roll)
    y = math.radians(yaw)

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
    Rz = np.array([
        [math.cos(y),-math.sin(y),0],
        [math.sin(y), math.cos(y),0],
        [0,0,1]
    ])
    return Rz @ Ry @ Rx

def parse_line(line):
    try:
        parts = line.strip().split(',')
        if len(parts)<3:
            return None,None,None
        return float(parts[0]), float(parts[1]), float(parts[2])
    except:
        return None,None,None

def init():
    global pyramid
    faces = [[vertices[i] for i in f] for f in faces_idx]
    pyramid = Poly3DCollection(faces, facecolors=face_colors, edgecolors='k', alpha=0.8)
    ax2.add_collection3d(pyramid)
    return (line_pitch,line_roll,line_yaw,pyramid)

def update(frame):
    raw = ser.readline().decode(errors='ignore')
    if not raw:
        return (line_pitch,line_roll,line_yaw,pyramid)
    pitch, roll, yaw = parse_line(raw)
    if pitch is None:
        return (line_pitch,line_roll,line_yaw,pyramid)

    pitch_buf.append(pitch)
    roll_buf.append(roll)
    yaw_buf.append(yaw)
    x_idx.append(len(x_idx)+1 if x_idx else 1)

    # Update time-series
    xs = list(range(len(x_idx)))
    line_pitch.set_data(xs,list(pitch_buf))
    line_roll.set_data(xs,list(roll_buf))
    line_yaw.set_data(xs,list(yaw_buf))
    ax1.set_xlim(max(0,len(xs)-WINDOW), max(WINDOW,len(xs)))

    # Rotate pyramid
    R = rotation_matrix(pitch, roll, yaw)
    rotated = np.dot(vertices, R.T)
    new_faces = [[rotated[i] for i in f] for f in faces_idx]
    pyramid.set_verts(new_faces)

    return (line_pitch,line_roll,line_yaw,pyramid)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=50, blit=False)
plt.tight_layout()
plt.show()
