import 'package:flutter/material.dart';

class Metrics {
  final double accuracy;
  final double precision;
  final double recall;
  final double f1Score;
  final double aucRoc;
  final List<Map<String, double>> rocCurve;

  const Metrics({
    required this.accuracy,
    required this.precision,
    required this.recall,
    required this.f1Score,
    required this.aucRoc,
    required this.rocCurve,
  });

  factory Metrics.fromJson(Map<String, dynamic> json) {
    return Metrics(
      accuracy: (json['accuracy'] as num).toDouble(),
      precision: (json['precision'] as num).toDouble(),
      recall: (json['recall'] as num).toDouble(),
      f1Score: (json['f1_score'] as num).toDouble(),
      aucRoc: (json['auc_roc'] as num).toDouble(),
      rocCurve: (json['roc_curve'] as List<dynamic>)
          .map(
            (point) => {
              "fpr": (point['fpr'] as num).toDouble(),
              "tpr": (point['tpr'] as num).toDouble(),
            },
          )
          .toList(),
    );
  }
}

class MetricsDashboard extends StatelessWidget {
  final Metrics metrics;

  const MetricsDashboard({super.key, required this.metrics});

  @override
  Widget build(BuildContext context) {
    // Lista definita per mappare le chiavi del JSON a etichette leggibili e colori
    final List<Map<String, dynamic>> displayData = [
      {
        'label': 'Accuracy',
        'value': metrics.accuracy,
        'icon': Icons.check_circle_outline,
        'color': Colors.blue,
      },
      {
        'label': 'Precision',
        'value': metrics.precision,
        'icon': Icons.trending_up,
        'color': Colors.green,
      },
      {
        'label': 'Recall',
        'value': metrics.recall,
        'icon': Icons.history,
        'color': Colors.orange,
      },
      {
        'label': 'F1 Score',
        'value': metrics.f1Score,
        'icon': Icons.functions,
        'color': Colors.purple,
      },
      {
        'label': 'AUC ROC',
        'value': metrics.aucRoc,
        'icon': Icons.show_chart,
        'color': Colors.red,
      },
    ];

    return GridView.builder(
      shrinkWrap: true,
      physics:
          const NeverScrollableScrollPhysics(), // Importante per non andare in conflitto con SingleChildScrollView
      itemCount: displayData.length,
      gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
        maxCrossAxisExtent: 200,
        mainAxisExtent:
            110, // <--- FIX: Forza un'altezza fissa per ogni riga di card
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
      ),
      itemBuilder: (context, index) => _buildMetricCard(displayData[index]),
    );
  }

  Widget _buildMetricCard(Map<String, dynamic> item) {
    final double value = (item['value'] as num).toDouble();
    final String displayValue = "${(value * 100).toStringAsFixed(1)}%";

    return Card(
      elevation: 4,
      // Scuriamo leggermente la card per farla stare bene nel tuo sfondo 0xFF1A1A1A
      color: const Color(0xFF252525),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        // Riduciamo il padding interno (da 16 a 8) per dare respiro ai widget
        padding: const EdgeInsets.all(8.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment
              .spaceEvenly, // Distribuisce lo spazio in modo intelligente
          children: [
            Icon(
              item['icon'],
              color: item['color'],
              size: 24,
            ), // Ridotto leggermente (da 30 a 24)

            Text(
              item['label'].toUpperCase(),
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 10, // Font più piccolo per l'etichetta
                color: Colors.white70,
              ),
            ),

            // Il trucco magico: FittedBox impedisce l'overflow del testo
            FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                displayValue,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: item['color'],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
