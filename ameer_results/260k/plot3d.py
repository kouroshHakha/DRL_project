import matplotlib.pyplot as plt
import pickle
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

rspecs= pickle.load(open( "obs_reached", "rb" ))
nspecs= pickle.load(open("obs_nreached","rb"))
x_r =[]
y_r =[]
z_r =[]
for spec in rspecs:
    x_r.append(spec[0])
    y_r.append(spec[1])
    z_r.append(spec[3])


x_nr =[]
y_nr =[]
z_nr =[]
for spec in nspecs:
    x_nr.append(spec[0])
    y_nr.append(spec[1])
    z_nr.append(spec[3])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_r,y_r,z_r)
ax.scatter(x_nr,y_nr,z_nr)

plt.show()
#ax = plt.axes(projection='3d')
#ax.scatter(x_r, y_r, z_r, c=z_r, cmap='viridis', linewidth=0.5);
#Axes3D.scatter(x_r, y_r z_r)
