# Detaillierte Dokumentation der Code-Änderungen und FEM-Implementierung

Dieses Dokument beschreibt exakt alle durchgeführten Änderungen am ursprünglichen Python-Script `HA3.py`, um das statisch unterbestimmte Stabwerksmodell (Mechanismus) stabil und physikalisch korrekt zu berechnen.

## 1. Numerische Präzision & Datentypen
**Ort:** Global (Initialisierung aller Tensoren)
*   **Original:** Implizite Nutzung von `float32` (Standard in PyTorch).
*   **Änderung:** Explizite Umstellung aller Tensoren (`x`, `u`, `K`, `loadsteps`, etc.) auf `dtype=torch.double` (float64).
*   **Grund:** Das System ist statisch empfindlich (Mechanismus). `float32` besitzt nicht genügend signifikante Stellen, um die fast singuläre Steifigkeitsmatrix stabil zu invertieren. Dies verhinderte "Not a Number" (NaN) Fehler.

## 2. Interpolation der Lastfunktionen
**Ort:** Zeile ~105
*   **Original:** `mode='bilinear'` (eigentlich für Bilder gedacht).
*   **Änderung:**
    ```python
    loadsteps_interpolated = torch.nn.functional.interpolate(..., mode='linear', align_corners=True)
    ```
*   **Grund:** Die ursprüngliche Interpolation führte zu Sprüngen oder Null-Werten zu Beginn der Zeitreihe. Die lineare Interpolation garantiert einen glatten Lastanstieg ($0 \to F_{max}$), was für das Newton-Verfahren essenziell ist.

## 3. Initialisierung der Randbedingungen
**Ort:** Zeile ~216 (Innerhalb der Zeitschleife, vor der Iteration)
*   **Original:** Fehlte komplett.
*   **Änderung:** Hinzugefügt:
    ```python
    u[drltDofs.view(-1).long()] = u_d[drltDofs.view(-1).long()]
    ```
*   **Grund:** Die vorgeschriebenen Verschiebungen müssen **vor** der ersten Kraftberechnung im Vektor `u` stehen, damit das System konsistent startet.

## 4. Korrektur der Freiheitsgrad-Zuordnung (DOF-Mapping)
**Ort:** Zeile ~224 (Innerhalb der Elementschleife)
*   **Original:** Fehlerhafte Index-Berechnung (`node*ndf-1:node*ndf`), die zu Dimensionierungsfehlern führte.
*   **Änderung:** Saubere Listen-basierte Zuweisung:
    ```python
    gdof_list = []
    for node in range(nen):
        start_dof = conn[e, node].item() * ndf
        gdof_list.extend([start_dof, start_dof+1])
    gdof = torch.tensor(gdof_list, ...)
    ```
*   **Grund:** Korrektes "Einsortieren" der lokalen Elementsteifigkeiten in die globale Matrix `K`.

## 5. Element-Formulierung (Co-Rotational Formulation)
**Ort:** Zeile ~235 - 280 (Kern der Elementschleife)
Dies ist die wichtigste physikalische Änderung.
*   **Original:** Linearisierter Ansatz (`B`-Matrix), der Verschiebungen direkt als Verzerrungen interpretiert.
    *   *Problem:* Bei einer Drehung des Stabes (ohne Dehnung) ändern sich die Koordinaten. Der lineare Ansatz interpretiert dies fälschlicherweise als riesige Dehnung $\to$ Explosion der Kräfte.
*   **Änderung:** Implementierung der **Co-Rotational Formulation**:
    1.  **Geometrie-Update:** `xe = xe_initial + ue_curr` (Rechnen mit verformter Lage).
    2.  **Exakte Dehnung:** `eps = (L_aktuell - L_0) / L_0`. Dies filtert Starrkörperdrehungen heraus.
    3.  **Interne Kräfte:** Vektor zeigt immer exakt in die *aktuelle* Stabrichtung `n`.
    4.  **Tangentensteifigkeit:**
        *   `Km` (Materialanteil): Widerstand gegen Dehnung ($EA/L$).
        *   `Kg` (Geometrischer Anteil): Widerstand gegen Querdrehung bei Zugkraft ("Seileffekt").
        *   `Ke = Km + Kg`.
*   **Grund:** Nur so können große Rotationen (wie bei diesem Mechanismus) korrekt berechnet werden.

## 6. Solver-Implementierung (Newton-Raphson)
**Ort:** Zeile ~290 (Die `# TODO` Blöcke)
*   **Original:** Leere Platzhalter.
*   **Implementierung:**
    1.  **Residuum:** `rsd_F = fext - fint` (Ungleichgewichtskräfte).
    2.  **Konvergenz-Check:** `if norm(rsd) < tol: break`.
    3.  **Dirichlet-Einbau:** Modifikation von `K` für den Solver:
        *   Zeilen/Spalten der fixierten Knoten auf 0 setzen.
        *   Diagonalelement auf 1 setzen.
        *   Rechte Seite (`rhs`) an diesen Stellen auf 0 setzen.
    4.  **Stabilisierung:** `K_solve.diagonal().add_(1e-2)`. Verhindert Singularität bei beweglichen Teilen (Mechanismen).
    5.  **Lösen:** `du = solve(K, rhs)` und Update `u += du`.

## 7. Berechnung der Reaktionskräfte
**Ort:** Zeile ~310
*   **Original:** `fext = K_tilde * u` (Elementweise Multiplikation).
*   **Änderung:** `fext = torch.matmul(K_tilde, u)` (Matrix-Vektor-Produkt).
*   **Grund:** Mathematisch notwendig für die Berechnung $F = K \cdot u$.

## 8. Visualisierung & Interaktivität
**Ort:** Ab Zeile 120 und Ende der Schleife
*   **Änderungen:**
    1.  **Interaktiver Modus:** `plt.ion()` statt blockierendem `plt.show()`.
    2.  **Fester Rahmen:** Fixierung von `xlim/ylim` und `cax` (Colorbar-Achse), um das "Schrumpfen" des Plots zu verhindern.
    3.  **Effizienz:** Nutzen von `line.set_data()` statt `ax.clear()`, damit der Plot flüssig läuft.
    4.  **Spannungs-Anzeige:** Einfärben der Stäbe nach MPa-Wert (Jet-Colormap).
    5.  **Tooltip:** Hover-Funktion, die beim Überfahren mit der Maus die Element-ID und die Spannung in MPa anzeigt.

---
**Fazit:** Der Code wurde von einem linearen Skelett zu einem vollwertigen, geometrisch nichtlinearen FEM-Solver erweitert, der speziell für große Verformungen und kinematische Mechanismen ausgelegt ist.
