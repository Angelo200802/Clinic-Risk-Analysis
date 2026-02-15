import 'package:flutter/material.dart';

class CorrelationMatrixWidget extends StatelessWidget {
  final List<String> labels; // Nomi variabili: ["HR", "BMI", "Age", ...]
  final List<List<double>> matrix; // I valori della matrice

  const CorrelationMatrixWidget({
    super.key,
    required this.labels,
    required this.matrix,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Calcoliamo la dimensione della cella in base alla larghezza disponibile
        double cellSize = (constraints.maxWidth - 60) / labels.length;
        cellSize = cellSize.clamp(
          40.0,
          80.0,
        ); // Evitiamo celle troppo piccole o giganti

        return Column(
          children: [
            // Header orizzontale
            Row(
              children: [
                const SizedBox(width: 60), // Spazio per le etichette verticali
                ...labels.map(
                  (label) => SizedBox(
                    width: cellSize,
                    child: Text(
                      label,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white54,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            // Righe della matrice
            ...List.generate(matrix.length, (rowIdx) {
              return Row(
                children: [
                  // Etichetta verticale
                  SizedBox(
                    width: 60,
                    child: Text(
                      labels[rowIdx],
                      textAlign: TextAlign.right,
                      style: const TextStyle(
                        color: Colors.white54,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  // Celle di calore
                  ...List.generate(matrix[rowIdx].length, (colIdx) {
                    double value = matrix[rowIdx][colIdx];
                    return _CorrelationCell(value: value, size: cellSize);
                  }),
                ],
              );
            }),
          ],
        );
      },
    );
  }
}

class _CorrelationCell extends StatelessWidget {
  final double value;
  final double size;

  const _CorrelationCell({required this.value, required this.size});

  @override
  Widget build(BuildContext context) {
    // Colore basato sul valore: Blu per positivo, Rosso per negativo
    Color cellColor = value >= 0
        ? Colors.blueAccent.withOpacity(value.clamp(0.1, 1.0))
        : Colors.redAccent.withOpacity(value.abs().clamp(0.1, 1.0));

    return Container(
      width: size,
      height: size,
      margin: const EdgeInsets.all(1),
      decoration: BoxDecoration(
        color: cellColor,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Center(
        child: Text(
          value.toStringAsFixed(2),
          style: TextStyle(
            color: value.abs() > 0.5 ? Colors.white : Colors.white70,
            fontSize: size < 50 ? 8 : 10,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}
