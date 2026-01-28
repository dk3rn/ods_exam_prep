import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Daten vorbereiten
x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)

# Figur erstellen
fig = plt.figure(figsize=(14, 6))

# 1. Subplot: 3D Oberfläche
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=True, alpha=0.8)
ax1.set_title('3D-Oberfläche der Himmelblau-Funktion')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# 2. Subplot: Konturplot
ax2 = fig.add_subplot(1, 2, 2)
# Logarithmische Skalierung der Konturen, um die Minima besser zu sehen
cp = ax2.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap=cm.viridis)
fig.colorbar(cp, ax=ax2)
ax2.set_title('Konturplot (Draufsicht)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()
plt.show()