import numpy as np
import time
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Definition der Fitnessfunktion (Himmelblau-Funktion)
# ---------------------------------------------------------
def fitness_function(pos):
    """
    Berechnet den Fitnesswert basierend auf der Himmelblau-Funktion.
    f(x, y) = (x^2 + y + 11)^2 + (x + y^2 - 7)^2

    :param pos: Ein Array oder Liste mit [x, y] Koordinaten
    :return: Der Funktionswert (soll minimiert werden)
    """
    x = pos[0]
    y = pos[1]
    return (x ** 2 + y + 11) ** 2 + (x + y ** 2 - 7) ** 2


# ---------------------------------------------------------
# 2. Particle Swarm Optimization (PSO) Klasse
# ---------------------------------------------------------
class ParticleSwarmOptimizer:
    def __init__(self, n_particles, iterations, bounds):
        self.n_particles = n_particles
        self.iterations = iterations
        self.bounds = bounds  # [(min_x, max_x), (min_y, max_y)]

        # --- BEGRÜNDUNG DER EINSTELLPARAMETER ---
        # w (Inertia Weight): 0.7
        # Ein Wert < 1 sorgt dafür, dass die Geschwindigkeit über die Zeit abnimmt.
        # Dies erlaubt anfangs Exploration (Suche im weiten Raum) und später
        # Exploitation (Feintuning um das Minimum).
        self.w = 0.7

        # c1 (Cognitive Component): 1.4
        # Bestimmt, wie stark das Teilchen zu seiner eigenen besten bekannten Position (p_best) zieht.
        # Ein moderater Wert fördert die individuelle Erfahrung.
        self.c1 = 1.4

        # c2 (Social Component): 1.4
        # Bestimmt, wie stark das Teilchen zur besten Position des gesamten Schwarms (g_best) zieht.
        # Werte für c1 und c2 werden oft gleich gewählt (Summe oft ca. 4.0 in älterer Literatur,
        # aber 1.4/1.4 mit w=0.7 ist eine stabile Standardkonfiguration).
        self.c2 = 1.4

        # Initialisierung der Partikel
        # Positionen zufällig im Wertebereich -5 bis +5
        self.positions = np.random.uniform(bounds[0][0], bounds[0][1], (n_particles, 2))

        # Geschwindigkeiten initial auf 0 setzen
        self.velocities = np.zeros((n_particles, 2))

        # Beste bekannte Position jedes einzelnen Partikels (p_best)
        self.p_best_pos = self.positions.copy()
        self.p_best_val = np.array([fitness_function(p) for p in self.positions])

        # Beste bekannte Position des gesamten Schwarms (g_best)
        min_idx = np.argmin(self.p_best_val)
        self.g_best_pos = self.p_best_pos[min_idx].copy()
        self.g_best_val = self.p_best_val[min_idx]

    def optimize(self):
        start_time = time.time()

        # Speicher für Historie (nur für Visualisierung)
        history = []

        for i in range(self.iterations):
            history.append(self.positions.copy())

            # Zufallsfaktoren r1 und r2 (Stochastische Komponente)
            # Dimension: (n_particles, 2) damit x und y unterschiedlich gewichtet werden können
            r1 = np.random.rand(self.n_particles, 2)
            r2 = np.random.rand(self.n_particles, 2)

            # --- UPDATE DER GESCHWINDIGKEIT ---
            # v_neu = w * v_alt + c1 * r1 * (p_best - pos) + c2 * r2 * (g_best - pos)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.p_best_pos - self.positions) +
                               self.c2 * r2 * (self.g_best_pos - self.positions))

            # --- UPDATE DER POSITION ---
            # pos_neu = pos_alt + v_neu
            self.positions = self.positions + self.velocities

            # Begrenzung auf den Suchraum (Clamping)
            # Wenn Partikel rausfliegen, werden sie an den Rand gesetzt
            self.positions[:, 0] = np.clip(self.positions[:, 0], self.bounds[0][0], self.bounds[0][1])
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.bounds[1][0], self.bounds[1][1])

            # --- EVALUIERUNG ---
            current_fitness = np.array([fitness_function(p) for p in self.positions])

            # Update Personal Best (p_best)
            improved_indices = np.where(current_fitness < self.p_best_val)
            self.p_best_pos[improved_indices] = self.positions[improved_indices]
            self.p_best_val[improved_indices] = current_fitness[improved_indices]

            # Update Global Best (g_best)
            if np.min(current_fitness) < self.g_best_val:
                min_idx = np.argmin(current_fitness)
                self.g_best_pos = self.positions[min_idx].copy()
                self.g_best_val = current_fitness[min_idx]

        end_time = time.time()
        duration = end_time - start_time

        return self.g_best_pos, self.g_best_val, duration, history


# ---------------------------------------------------------
# 3. Ausführung und Diskussion
# ---------------------------------------------------------

# Parameter Setup
# Wertebereich für x und y: -5 bis +5
bounds = [(-5, 5), (-5, 5)]
n_particles = 30  # 30 Partikel sind meist ausreichend für 2D-Probleme
iterations = 100  # 100 Iterationen reichen oft für Konvergenz bei dieser Funktion

print("Start Particle Swarm Optimization...")
pso = ParticleSwarmOptimizer(n_particles, iterations, bounds)
best_pos, best_val, duration, history = pso.optimize()

print("-" * 40)
print(f"Laufzeit PSO: {duration:.6f} Sekunden")
print(f"Gefundenes Minimum bei x, y: {best_pos}")
print(f"Fitnesswert (Funktionswert): {best_val:.6f}")
print("-" * 40)

# --- DISKUSSION DER ERGEBNISSE ---
# Die Himmelblau-Funktion hat 4 globale Minima, alle mit f(x,y) = 0.
# Die bekannten Minima liegen bei ca.:
# 1. (3.0, 2.0)
# 2. (-2.805118, 3.131312)
# 3. (-3.779310, -3.283186)
# 4. (3.584428, -1.848126)
#
# Da PSO ein stochastischer Algorithmus ist, wird er eines dieser 4 Minima finden.
# Welches gefunden wird, hängt von der zufälligen Initialisierung ab.
# Wenn der Fitnesswert nahe 0 liegt (z.B. < 1e-5), war die Optimierung erfolgreich.

# ---------------------------------------------------------
# 4. (Optional) Visualisierung des Ergebnisses
# ---------------------------------------------------------
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = (X ** 2 + Y + 11) ** 2 + (X + Y ** 2 - 7) ** 2

plt.figure(figsize=(10, 8))
# Konturplot der Funktion
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Fitness f(x,y)')

# Plot der Partikel (letzte Iteration)
# Startpositionen (optional)
# plt.scatter(history[0][:,0], history[0][:,1], color='white', marker='x', label='Start')
# Endpositionen
plt.scatter(history[-1][:, 0], history[-1][:, 1], color='red', marker='o', label='Endpositionen')
plt.scatter(best_pos[0], best_pos[1], color='yellow', marker='*', s=200, edgecolors='black', label='Global Best')

plt.title('Himmelblau-Funktion - PSO Optimierung')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()