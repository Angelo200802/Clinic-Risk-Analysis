import 'package:flutter/material.dart';

class ConfusionMatrixWidget extends StatelessWidget {
  final Map<String, dynamic> data;

  const ConfusionMatrixWidget({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    // Estraiamo i valori in modo sicuro
    final int tp = data['TP'] ?? 0;
    final int tn = data['TN'] ?? 0;
    final int fp = data['FP'] ?? 0;
    final int fn = data['FN'] ?? 0;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          children: [
            _buildCell(
              "TRUE POSITIVE",
              tp,
              Colors.greenAccent,
              "Correct Prediction",
            ),
            const SizedBox(width: 10),
            _buildCell("FALSE POSITIVE", fp, Colors.redAccent, "Type I Error"),
          ],
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            _buildCell("FALSE NEGATIVE", fn, Colors.redAccent, "Type II Error"),
            const SizedBox(width: 10),
            _buildCell(
              "TRUE NEGATIVE",
              tn,
              Colors.greenAccent,
              "Correct Rejection",
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildCell(String title, int value, Color color, String sub) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: TextStyle(
                color: color,
                fontSize: 9,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              value.toString(),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              sub,
              style: const TextStyle(color: Colors.white24, fontSize: 8),
            ),
          ],
        ),
      ),
    );
  }
}
