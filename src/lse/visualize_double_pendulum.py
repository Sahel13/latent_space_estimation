import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

import jax.random as random

from data_generator import DoublePendulum

key = random.PRNGKey(100)
pend = DoublePendulum()
key, subkey = random.split(key)
train_data = pend.get_trajectory(subkey)

fig = plt.figure(figsize=(5, 4))
L = 2.1
history_len = 500
dt = 0.01
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

x1, y1, x2, y2 = train_data[4:8]

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(x1), interval=dt, blit=True)
plt.show()