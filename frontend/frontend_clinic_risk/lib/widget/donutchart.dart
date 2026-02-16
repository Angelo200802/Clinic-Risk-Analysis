import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class DetailedEnsemblePieChart extends StatelessWidget {
  final List<Map<String, dynamic>> data;

  const DetailedEnsemblePieChart({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Text(
          "Intersezione Correttezza Modelli",
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 30),
        SizedBox(
          height: 280,
          child: PieChart(
            PieChartData(
              sectionsSpace: 2,
              centerSpaceRadius: 50,
              sections: _buildDetailedSections(),
            ),
          ),
        ),
        const SizedBox(height: 20),
        // Legenda dettagliata per nomi combinazione
        _buildDetailedLegend(),
      ],
    );
  }

  List<PieChartSectionData> _buildDetailedSections() {
    return data.map((item) {
      final String label = item['combination'];
      final double value = item['count'].toDouble();
      final bool isFullConsensus = label.split('+').length == 3;

      return PieChartSectionData(
        color: _getSpecificColor(label),
        value: value,
        title: value > 5000
            ? label
            : '', // Mostra etichetta solo se la fetta è grande
        radius: isFullConsensus
            ? 35
            : 25, // Risalta la fetta del consenso totale
        titleStyle: const TextStyle(
          fontSize: 10,
          fontWeight: FontWeight.bold,
          color: Colors.white,
        ),
      );
    }).toList();
  }

  Widget _buildDetailedLegend() {
    return Wrap(
      spacing: 10,
      runSpacing: 10,
      children: data.map((item) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: _getSpecificColor(item['combination']),
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 4),
            Text(
              "${item['combination']} (${item['count']})",
              style: const TextStyle(color: Colors.white70, fontSize: 11),
            ),
          ],
        );
      }).toList(),
    );
  }

  Color _getSpecificColor(String comb) {
    if (comb == "LR+MLP+NB") return Colors.redAccent;
    if (comb == "LR+MLP") return Colors.blueAccent;
    if (comb == "LR+NB") return Colors.amber;
    if (comb == "MLP+NB") return Colors.deepOrangeAccent;
    if (comb == "LR") return Colors.pinkAccent;
    if (comb == "MLP") return Colors.deepPurple;
    if (comb == "NB") return Colors.brown;
    return Colors.greenAccent; // Per "Nessuno"
  }
}
