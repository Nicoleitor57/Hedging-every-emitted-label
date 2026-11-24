"""Programación del LR y parada temprana del entrenamiento conjunto.

Por qué (curvas de val de los 12 arms de una tanda de desarrollo):
  - no había programación del LR en ninguna parte: AdamW con LR constante las 8 épocas;
  - el mejor val strict cae en la época 8 o 7 en 11/12 arms, así que no se aplanó;
  - la val salta hasta −4.6 pt de una época a la siguiente (E_supcon1_open__s14 68.26 → 64.32
    → 68.74), lo esperable con LR constante, y elegir checkpoint entre esos saltos agrega ruido.

La parada temprana hace que el horizonte NO se conozca de antemano, así que la programación es de
HORIZONTE LIBRE ('plateau'): por ÉPOCA, baja el LR de todos los grupos ×factor tras `patience`
épocas seguidas sin mejorar la val. No depende de dónde corte la parada.
Implementado aquí y NO con torch.optim.lr_scheduler.ReduceLROnPlateau a propósito: allí
`patience=1` baja recién tras DOS épocas malas y la mejora es relativa (threshold 1e-4), así
que "paciencia" y "mejora" significarían otra cosa que en la parada temprana y en el guardado
del checkpoint. Aquí: mejora = estrictamente mayor; N épocas sin mejorar = actuar.

La parada temprana y la meseta usan el MISMO criterio que ya elige el checkpoint (val e2e
strict), así el checkpoint evaluado en test sigue siendo el mejor de las épocas corridas.
"""

class PlateauLR:
    """Baja el LR de TODOS los grupos ×`factor` tras `patience` épocas seguidas sin mejorar.

    `step(metric)` se llama UNA vez por época, después de evaluar val (mayor es mejor).
    Mejora = estrictamente mayor que el mejor visto, igual que el guardado del checkpoint.
    Tras bajar, el contador vuelve a 0: con patience=1 cada época sin mejora baja el LR.
    """

    def __init__(self, optimizer, factor=0.5, patience=1):
        if not 0.0 < factor < 1.0:
            raise ValueError(f'factor debe estar en (0,1), llegó {factor}')
        if patience < 1:
            raise ValueError(f'patience debe ser >= 1, llegó {patience}')
        self.optimizer, self.factor, self.patience = optimizer, factor, patience
        self.best, self.bad, self.n_reductions = None, 0, 0

    def step(self, metric):
        """-> True si en esta llamada bajó el LR."""
        if self.best is None or metric > self.best:
            self.best, self.bad = metric, 0
            return False
        self.bad += 1
        if self.bad < self.patience:
            return False
        for g in self.optimizer.param_groups:
            g['lr'] *= self.factor
        self.bad = 0
        self.n_reductions += 1
        return True


def should_stop(epochs_done, best_epoch, patience, min_epochs):
    """True si hay que cortar DESPUÉS de la época `epochs_done` (1-indexada).

    Corta cuando van `patience` épocas seguidas sin mejorar el mejor val, y nunca antes de
    completar `min_epochs`. patience <= 0 = apagado.
    """
    if patience <= 0:
        return False
    return epochs_done >= min_epochs and (epochs_done - best_epoch) >= patience
