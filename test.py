import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

matplotlib.rcParams["mathtext.fontset"]="cm"

mean = 1
x = np.arange(-2, 4, 0.01)
y = norm.pdf(x, loc=mean)

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(x, y)
ax1.axvline(x=mean, color="r")
ax1.fill_between(x, np.zeros(len(y)), y, where=x<=0, alpha=0.5)

ax1.annotate("$\\Phi(-\\dfrac{|m_n(x)|}{\\sigma_n(x)})$", xy=(-0.5, 0.1), xytext=(-50, 50), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
ax1.annotate("$m_n(x)>0$", xy=(mean, 0.1), xytext=(0, 0), textcoords="offset points")

ax1.spines["left"].set_position("zero")
ax1.spines["right"].set_visible(False)
ax1.yaxis.tick_left()
ax1.spines["bottom"].set_position("zero")
ax1.spines["top"].set_visible(False)
ax1.xaxis.tick_bottom()
ax1.set_aspect(5)

ax1.set_yticks([])
ax1.set_xticks([0])

mean=-1
x = np.arange(-4, 2, 0.01)
y = norm.pdf(x, loc=mean)
ax2.plot(x, y)
ax2.axvline(x=mean, color="r")
ax2.fill_between(x, np.zeros(len(y)), y, where=x>=0, alpha=0.5)

ax2.annotate("$\\Phi(-\\dfrac{|m_n(x)|}{\\sigma_n(x)})$", xy=(0.5, 0.1), xytext=(50, 50), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
ax2.annotate("$m_n(x)<0$", xy=(mean, 0.1), xytext=(0, 0), textcoords="offset points")

ax2.spines["left"].set_position("zero")
ax2.spines["right"].set_visible(False)
ax2.yaxis.tick_left()
ax2.spines["bottom"].set_position("zero")
ax2.spines["top"].set_visible(False)
ax2.xaxis.tick_bottom()
ax2.set_aspect(5)

ax2.set_yticks([])
ax2.set_xticks([0])
plt.show()