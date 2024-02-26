import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator #minor grid
import dynamics as dyn
from svgpathtools import svg2paths
from svgpath2mpl import parse_path

################################
# Parameters
################################
dt = dyn.dt
tf = 20
TT = int(tf/dt)
tt_hor = np.linspace(0,tf,TT)

time = np.arange(len(tt_hor))*dt

#scale factor
scale_factor = 5

xx_result = np.load('x_opt_task3.npy') 
xx_ref = np.load('x_ref.npy') 

#background
road_im = plt.imread("empty-straight-road.jpg")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, autoscale_on=False, xlim=(min(xx_result[0]) -10, max(xx_result[0])+10), ylim=(min(xx_result[1])-15, max(xx_result[1])+30))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
im = ax.imshow(road_im, extent=[min(xx_result[0]) -10, max(xx_result[0])+10, min(xx_result[1])-15, max(xx_result[1])+30])


time_template = 't = %.1f s'
time_text = ax.text(0.45, 0.75, '', transform=ax.transAxes, c='w')
fig.gca().set_aspect('equal', adjustable='box')

#racing marker creation
racing_car_path, attributes = svg2paths('racing_car.svg')
racing_car_marker = parse_path(attributes[0]['d'])
racing_car_marker.vertices -= racing_car_marker.vertices.mean(axis=0)

#optimal
point0, = ax.plot([], [], marker=racing_car_marker, c='m', markersize=60, label='Optimal')

#reference
ax.plot(xx_ref[0], xx_ref[1]*scale_factor, lw=2, c='w', dashes=[1, 1], label="Reference")

# Subplot
left, bottom, width, height = [0.64, 0.13, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.xaxis.set_major_locator(MultipleLocator(3))
ax2.yaxis.set_major_locator(MultipleLocator(5))
ax2.set_xlabel("time (s)")
ax2.set_ylabel("$x_1$")

ax2.grid(which='both')
ax2.plot(time, xx_result[1],c='b')
ax2.plot(time, xx_ref[1], c='g', dashes=[1, 1])

point1, = ax2.plot([], [], 'o', lw=2, c='b')


def init():
    point0.set_data([], [])

    point1.set_data([], [])

    time_text.set_text('')
    return point0, time_text, point1


def animate(i):
    # Trajectory
    thisx0 = [(xx_result[0, i]), (xx_result[0, i])]
    thisy0 = [(xx_result[1, i])*scale_factor, (xx_result[1, i])*scale_factor]
    point0.set_data(thisx0, thisy0)

    point1.set_data(i*dt, xx_result[1, i])

    time_text.set_text(time_template % (i*dt))
    return point0, time_text, point1


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=TT, interval=-100, blit=True)
ax.legend(loc="upper left", markerscale=0.4)
  
plt.show()