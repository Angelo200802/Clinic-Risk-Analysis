import 'package:flutter/material.dart';

class DemographicStressMap extends StatelessWidget {
  final List<Map<String, dynamic>> data;

  const DemographicStressMap({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    // Estraiamo le decadi (chiavi del dizionario escluso 'BMI_Category')
    List<String> decades = data[0].keys
        .where((k) => k != "BMI_Category")
        .toList();
    decades.sort();

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Demographic Stress Map",
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Text(
            "Pressione Sistolica Media per BMI ed Età",
            style: TextStyle(color: Colors.white38, fontSize: 12),
          ),
          const SizedBox(height: 20),

          // Header delle colonne (Età)
          Row(
            children: [
              const SizedBox(width: 100), // Spazio per la label BMI
              ...decades.map(
                (d) => Expanded(
                  child: Center(
                    child: Text(
                      d.toString(),
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 10,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),

          // Righe della Heatmap
          ...data.map(
            (row) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                children: [
                  SizedBox(
                    width: 100,
                    child: Text(
                      row['BMI_Category'],
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                    ),
                  ),
                  ...decades.map((d) {
                    double val = (row[d] ?? 0).toDouble();
                    return Expanded(child: _buildHeatCell(val));
                  }),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeatCell(double value) {
    // Logica colore: più alta è la pressione, più è rosso
    // Basiamoci su range clinici: 110 (verde) -> 150 (rosso)
    Color cellColor = Color.lerp(
      Colors.tealAccent.withOpacity(0.2),
      Colors.redAccent.withOpacity(0.8),
      ((value - 110) / 40).clamp(0.0, 1.0),
    )!;

    return Container(
      height: 40,
      margin: const EdgeInsets.all(2),
      decoration: BoxDecoration(
        color: cellColor,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Center(
        child: Text(
          value > 0 ? value.toStringAsFixed(2) : "-",
          style: const TextStyle(
            color: Colors.white,
            fontSize: 10,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}
