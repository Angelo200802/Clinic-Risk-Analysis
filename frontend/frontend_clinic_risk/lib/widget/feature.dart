import 'package:flutter/material.dart';

class FeatureInsight {
  final String name;
  final double value; // Il valore attuale
  final double deltaPercentage;

  FeatureInsight({
    required this.name,
    required this.value,
    required this.deltaPercentage,
  });
}

class FeatureRow extends StatelessWidget {
  final FeatureInsight insight;

  const FeatureRow({super.key, required this.insight});

  @override
  Widget build(BuildContext context) {
    bool isIncreasing = insight.deltaPercentage > 0;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              // Nome della feature e Icona Trend
              Row(
                children: [
                  Icon(
                    isIncreasing ? Icons.arrow_upward : Icons.arrow_downward,
                    size: 14,
                    color: isIncreasing ? Colors.redAccent : Colors.blueAccent,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    insight.name,
                    style: const TextStyle(color: Colors.white70, fontSize: 13),
                  ),
                ],
              ),
              // Valore percentuale
              Text(
                "${isIncreasing ? '+' : ''}${insight.deltaPercentage.toStringAsFixed(1)}%",
                style: TextStyle(
                  color: isIncreasing ? Colors.redAccent : Colors.blueAccent,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                  fontFamily: 'monospace',
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          // Barra sottile di importanza (stile quello in foto)
          ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: LinearProgressIndicator(
              value: (insight.deltaPercentage.abs() / 20).clamp(
                0.1,
                1.0,
              ), // Normalizzato
              backgroundColor: Colors.white10,
              color: isIncreasing ? Colors.redAccent : Colors.blueAccent,
              minHeight: 3,
            ),
          ),
        ],
      ),
    );
  }
}

class FeaturePanel extends StatelessWidget {
  final List<FeatureInsight> insights;

  const FeaturePanel({super.key, required this.insights});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A).withOpacity(0.5),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                "FEATURE IMPORTANCE",
                style: TextStyle(
                  color: Colors.white54,
                  fontSize: 10,
                  letterSpacing: 1.5,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Icon(Icons.auto_graph, color: Colors.white24, size: 14),
            ],
          ),
          const Divider(color: Colors.white10, height: 20),
          // Lista delle feature
          ...insights.map((insight) => FeatureRow(insight: insight)),
        ],
      ),
    );
  }
}
