def berechne_mape_native(tatsaechlich, vorhersage):
    """
    Berechnet den MAPE (Mean Absolute Percentage Error).
    Behandelt Division durch Null, indem Nullen ignoriert oder gemeldet werden.
    """
    if len(tatsaechlich) != len(vorhersage):
        raise ValueError("Listen müssen gleich lang sein.")

    n = len(tatsaechlich)
    summe_prozentuale_fehler = 0
    gueltige_punkte = 0

    for i in range(n):
        wahr = tatsaechlich[i]
        prog = vorhersage[i]

        # Schutz vor Division durch Null
        if wahr == 0:
            print(f"Warnung: Wert an Index {i} ist 0. Wird übersprungen.")
            continue

        # 1. Absoluter Fehler
        abs_fehler = abs(wahr - prog)

        # 2. Relativer Fehler (Anteil am wahren Wert)
        rel_fehler = abs_fehler / abs(wahr)

        summe_prozentuale_fehler += rel_fehler
        gueltige_punkte += 1

    if gueltige_punkte == 0:
        return 0.0

    # 3. Durchschnitt bilden
    mape = summe_prozentuale_fehler / gueltige_punkte

    # 4. In Prozent umrechnen (* 100)
    return mape * 100


