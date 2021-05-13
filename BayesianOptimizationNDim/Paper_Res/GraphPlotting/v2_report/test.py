import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 17
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

x = np.linspace (2,8, 100)
y = -(np.sin(x) + np.sin(x*10/3))

plt.title('Global Optimisation')
plt.xlabel('X')
plt.ylabel('Output f(x)')
plt.plot(x,y, lw=3)
plt.show()
exit(0)
# plt.gca().fill_between(x,yrange_plus,yrange_minus)
# alpha =0.5
for i  in range (10 ,0 ,-1):
    print (i)
    fill1 = y + (yrange_plus - y) * i / 10
    fill2 = y - (y - yrange_minus) * i / 10
    # fill1 = yrang * i / 10
    # alpha = alpha /
    plt.gca().fill_between(x, fill1, fill2,
                           color="dimgray", alpha=0.1)
plt.show()
exit(0)

