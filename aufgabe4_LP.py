import numpy as np
import pandas as pd
from scipy.optimize import linprog


def solve_production_problem():
    # 1. ZIELFUNKTION
    # Wir wollen MAXIMIEREN: 45x1 + 53x2 + 37x3 - 12y1 - 12y2
    # Der Solver minimiert standardmäßig, daher drehen wir die Vorzeichen um (mal -1)
    # c = [x1, x2, x3, y1, y2]
    c = [-45, -53, -37, 12, 12]

    # 2. NEBENBEDINGUNGEN (Matrix A und Vektor b für A*x <= b)
    # x1=Typ1, x2=Typ2, x3=Typ3, y1=Überstunden_Spritz, y2=Überstunden_Fräse
    A = [
        # Spritzguß: 8x1 + 10x2 + 5x3 - y1 <= 3400
        [8, 10, 5, -1, 0],
        # Fräse: 4x1 + 7x2 + 6x3 - y2 <= 2700
        [4, 7, 6, 0, -1],
        # Überstunden-Limit: y1 + y2 <= 1000
        [0, 0, 0, 1, 1]
    ]
    b = [3400, 2700, 1000]

    # 3. GRENZEN (Bounds)
    # Mindestproduktion: 75 Stück pro Typ
    # Überstunden: min 0, max unendlich (bzw. durch Constraint begrenzt)
    x_bounds = (75, None)
    y_bounds = (0, None)
    bounds = [x_bounds, x_bounds, x_bounds, y_bounds, y_bounds]

    # 4. GANZZAHLIGKEIT (Integrality)
    # 1 = Ganzzahl (Reifen), 0 = Kontinuierlich (Stunden)
    integrality = [1, 1, 1, 0, 0]

    # 5. LÖSUNG BERECHNEN
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, integrality=integrality, method='highs')

    # 6. AUSGABE ALS TABELLE
    if res.success:
        # Daten für die Tabelle aufbereiten
        data = [
            ["Reifen Typ 1", f"{res.x[0]:.0f}", "Stück"],
            ["Reifen Typ 2", f"{res.x[1]:.0f}", "Stück"],
            ["Reifen Typ 3", f"{res.x[2]:.0f}", "Stück"],
            ["Überstunden Spritzguß", f"{res.x[3]:.2f}", "Stunden"],
            ["Überstunden Fräse", f"{res.x[4]:.2f}", "Stunden"],
            ["-----------------", "--------", "-------"],  # Trennlinie
            ["TOTAL GEWINN", f"{-res.fun:,.2f}", "Euro"]
        ]

        # DataFrame erstellen
        df = pd.DataFrame(data, columns=["Bezeichnung", "Wert", "Einheit"])

        # Ausgabe (ohne Index-Spalte links, damit es sauber aussieht)
        print("\n--- OPTIMALER BETRIEBSPLAN ---\n")
        print(df.to_string(index=False))
        print("\n")
    else:
        print("Keine optimale Lösung gefunden.")


if __name__ == "__main__":
    solve_production_problem()