import numpy as np
import matplotlib.pyplot as plt

# 1. Definition der Variablenbereiche
x = np.linspace(0, 1.2, 400) # Anteil Insektizid Alt

# 2. Berechnung der Grenzlinien
# Constraint A: 0.40x + 0.15y <= 0.36  => y <= (0.36 - 0.40x) / 0.15
y_limit_A = (0.36 - 0.40 * x) / 0.15

# Constraint B: 0.35x + 0.10y <= 0.28  => y <= (0.28 - 0.35x) / 0.10
y_limit_B = (0.28 - 0.35 * x) / 0.10

# Mischungsgerade: x + y = 1 => y = 1 - x
y_mix = 1 - x

# 3. Plot erstellen
plt.figure(figsize=(10, 8))

# Linien zeichnen
plt.plot(x, y_limit_A, label=r'Grenze Giftstoff A', color='red', linestyle='--')
plt.plot(x, y_limit_B, label=r'Grenze Giftstoff B', color='orange', linestyle='--')
plt.plot(x, y_mix, label=r'Mischung ($x+y=1$)', color='blue')

# Füllen des zulässigen Bereichs für die Grenzwerte (unterhalb beider Linien)
y_feasible_region = np.minimum(y_limit_A, y_limit_B)
plt.fill_between(x, 0, y_feasible_region, where=(y_feasible_region>=0), color='gray', alpha=0.1)

# Hervorheben des GÜLTIGEN Teils der Mischungsgeraden
# Wir müssen unterhalb der orangen und roten Linie bleiben.
# Schnittpunkt mit Orange (B): 0.35x + 0.10(1-x) = 0.28 => 0.25x = 0.18 => x = 0.72
# Schnittpunkt mit Rot (A): 0.40x + 0.15(1-x) = 0.36 => 0.25x = 0.21 => x = 0.84
# Der strengere Grenzwert ist x <= 0.72.
valid_range_x = np.linspace(0, 0.72, 100)
valid_range_y = 1 - valid_range_x
plt.plot(valid_range_x, valid_range_y, color='green', linewidth=5, label='Zulässige Mischung')

# Optimalen Punkt markieren (Maximales x, da x günstiger ist)
opt_x = 0.72
opt_y = 1 - opt_x
plt.scatter([opt_x], [opt_y], color='black', s=150, zorder=5, label='Optimaler Punkt')
plt.annotate(f' Optimum:\n {opt_x*100:.0f}% Alt\n {opt_y*100:.0f}% Neu', 
             xy=(opt_x, opt_y), xytext=(opt_x-0.2, opt_y+0.2),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Achsenbeschriftung und Layout
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.xlabel('Anteil Insektizid Alt (x)')
plt.ylabel('Anteil Insektizid Neu (y)')
plt.title('Lösung: Kostenminimierung unter Bundesvorgaben')
plt.legend()
plt.grid(True)

# Speichern und Anzeigen
plt.savefig('loesung_insektizid.png')
plt.show()