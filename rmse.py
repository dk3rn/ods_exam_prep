import math


def berechne_rmse_native(tatsaechlich, vorhersage):
    """
    Berechnet den RMSE ohne externe Bibliotheken (wie NumPy).

    Args:
        tatsaechlich (list): Liste der wahren Werte.
        vorhersage (list): Liste der vorhergesagten Werte.

    Returns:
        float: Der berechnete RMSE-Wert.
    """

    # 1. Überprüfung: Listen müssen gleich lang sein
    if len(tatsaechlich) != len(vorhersage):
        raise ValueError("Die Listen müssen die gleiche Länge haben.")

    n = len(tatsaechlich)
    summe_der_quadrate = 0

    # 2. Schleife durch alle Datenpunkte
    for i in range(n):
        # Differenz berechnen (Fehler)
        fehler = tatsaechlich[i] - vorhersage[i]

        # Fehler quadrieren
        fehler_quadriert = fehler ** 2

        # Zur Gesamtsumme addieren
        summe_der_quadrate += fehler_quadriert

    # 3. Mittelwert der quadrierten Fehler (Mean Squared Error - MSE)
    mse = summe_der_quadrate / n

    # 4. Wurzel ziehen (Root Mean Squared Error - RMSE)
    rmse = math.sqrt(mse)

    return rmse
