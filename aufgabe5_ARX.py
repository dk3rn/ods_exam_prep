import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Funktion zur Berechnung von RMSE und MAE
def evaluate_model(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

print("=== START SYSTEMIDENTIFIKATION ===")

# 0. DATEN LADEN
# ---------------------------------------------------------
u_df = pd.read_csv('u.csv', header=None)
y_df = pd.read_csv('y.csv', header=None)

u = u_df.values.flatten()
y = y_df.values.flatten()
N_raw = len(y)
print(f"Daten geladen. Anzahl Messwerte: {N_raw}")


# 1. GLÄTTUNG (Moving Average)
# ---------------------------------------------------------
# Anforderung: "Mindestens 8 Intervalleinheiten in die Zukunft"
# Wir nutzen ein zentriertes Fenster. Damit es 8 Schritte in die Zukunft schaut,
# muss der "Vorwärts-Anteil" des Fensters >= 8 sein.
# Fenstergröße N_win. Vorwärts = (N_win - 1) / 2
# (17 - 1) / 2 = 8. Wir wählen Fenstergröße 17.
print("\n--- SCHRITT 1: Glättung (Fenstergröße 17) ---")

window_size = 17
# rolling mit center=True schaut in Vergangenheit und Zukunft
u_smooth = pd.Series(u).rolling(window=window_size, center=True).mean()
y_smooth = pd.Series(y).rolling(window=window_size, center=True).mean()

# Ränder behandeln (NaNs durch Originalwerte oder bfill/ffill ersetzen)
# Hier nutzen wir Backfill/Forwardfill um die Datenlänge zu erhalten
u_smooth = u_smooth.fillna(method='bfill').fillna(method='ffill').values
y_smooth = y_smooth.fillna(method='bfill').fillna(method='ffill').values

# Visualisierung Schritt 1
plt.figure(figsize=(10, 4))
plt.plot(y, 'lightgray', label='Original')
plt.plot(y_smooth, 'b', label='Geglättet (MA 17)')
plt.title('Schritt 1: Glättung')
plt.legend()
plt.show()


# 2. SKALIERUNG (Standardisierung: Mean=0, Sigma=1)
# ---------------------------------------------------------
print("\n--- SCHRITT 2: Skalierung (Z-Score Normalisierung) ---")
u_mean, u_std = np.mean(u_smooth), np.std(u_smooth)
y_mean, y_std = np.mean(y_smooth), np.std(y_smooth)

u_scaled = (u_smooth - u_mean) / u_std
y_scaled = (y_smooth - y_mean) / y_std

print(f"u: Mean={u_mean:.2f}, Std={u_std:.2f} -> Skaliert: Mean={np.mean(u_scaled):.2f}, Std={np.std(u_scaled):.2f}")
print(f"y: Mean={y_mean:.2f}, Std={y_std:.2f} -> Skaliert: Mean={np.mean(y_scaled):.2f}, Std={np.std(y_scaled):.2f}")


# 3. ZEITLICHE VERSCHIEBUNG (Shift um 6 Einheiten)
# ---------------------------------------------------------
# "Verschiebung der Eingangs- und Ausgangsdaten um jeweils 6 Zeiteinheiten"
# Wir entfernen die ersten 6 Werte, um Start-Effekte oder Filter-Artefakte zu eliminieren.
print("\n--- SCHRITT 3: Zeitliche Verschiebung (Shift=6) ---")
shift_val = 6

u_shifted = u_scaled[shift_val:]
y_shifted = y_scaled[shift_val:]

# Zeitvektor anpassen
k_time = np.arange(len(y_shifted))

print(f"Daten nach Shift: {len(y_shifted)} Werte (Start bei Index {shift_val})")


# 4. ARX-MODELLERSTELLUNG
# ---------------------------------------------------------
print("\n--- SCHRITT 4: ARX-Modellierung ---")

# Aufteilung in Training (60%) und Test (40%)
split_idx = int(len(y_shifted) * 0.6)

u_train = u_shifted[:split_idx]
y_train = y_shifted[:split_idx]
u_test = u_shifted[split_idx:]
y_test = y_shifted[split_idx:]

# Erstellung der Regressor-Matrix für LS-Schätzung (Modell 1. Ordnung)
# y(k) = -a1*y(k-1) + b1*u(k-1)
# Wir nutzen die Trainingsdaten
y_target_train = y_train[1:]
Phi_train = np.column_stack((-y_train[:-1], u_train[:-1]))

# Parameterschätzung (Least Squares)
Theta = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ y_target_train
a1, b1 = Theta[0], Theta[1]

# Anpassung des Verstärkungsfaktors (falls notwendig)
# Hier nehmen wir die LS-Schätzung direkt.
print(f"Identifizierte Parameter (Trainingsdaten):")
print(f"a1 = {a1:.4f}")
print(f"b1 = {b1:.4f}")
print(f"Modellgleichung: y(k) = {-a1:.4f} * y(k-1) + {b1:.4f} * u(k-1)")


# 5. EVALUIERUNG (RMSE & Absoluter Fehler)
# ---------------------------------------------------------
print("\n--- SCHRITT 5: Evaluierung (RMSE & MAE) ---")

# Funktion zur Simulation des Modells
def simulate_arx(u_in, y_start, N_sim, a, b):
    y_sim = np.zeros(N_sim)
    y_sim[0] = y_start
    for k in range(1, N_sim):
        y_sim[k] = -a * y_sim[k-1] + b * u_in[k-1]
    return y_sim

# Simulation auf Trainingsdaten
y_pred_train = simulate_arx(u_train, y_train[0], len(y_train), a1, b1)

# Simulation auf Testdaten (Cross-Validation)
y_pred_test = simulate_arx(u_test, y_test[0], len(y_test), a1, b1)

# Fehlerberechnung
rmse_train, mae_train = evaluate_model(y_train, y_pred_train, "Train")
rmse_test, mae_test = evaluate_model(y_test, y_pred_test, "Test")

print(f"TRAINING SET -> RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
print(f"TEST SET     -> RMSE: {rmse_test:.4f},  MAE: {mae_test:.4f}")

# Plot der Ergebnisse
plt.figure(figsize=(12, 6))

# Plot Training
plt.subplot(1, 2, 1)
plt.plot(y_train, 'k-', alpha=0.6, label='Real (Train)')
plt.plot(y_pred_train, 'r--', label='Modell (Train)')
plt.title(f'Training (RMSE={rmse_train:.3f})')
plt.xlabel('Zeit k')
plt.ylabel('y (skaliert)')
plt.legend()
plt.grid(True)

# Plot Test
plt.subplot(1, 2, 2)
plt.plot(y_test, 'k-', alpha=0.6, label='Real (Test)')
plt.plot(y_pred_test, 'g--', label='Modell (Test)')
plt.title(f'Test (RMSE={rmse_test:.3f})')
plt.xlabel('Zeit k')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()