import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class DynamicRiskBarChart extends StatefulWidget {
  final List<dynamic> rawData;
  final String categoryKey;

  const DynamicRiskBarChart({
    super.key,
    required this.rawData,
    required this.categoryKey,
  });

  @override
  State<DynamicRiskBarChart> createState() => _DynamicRiskBarChartState();
}

class _DynamicRiskBarChartState extends State<DynamicRiskBarChart> {
  // Flag per controllare l'inizio dell'animazione
  bool _isLoaded = false;

  @override
  void initState() {
    super.initState();
    // Facciamo partire l'animazione dopo il primo frame
    Future.delayed(const Duration(milliseconds: 50), () {
      if (mounted) setState(() => _isLoaded = true);
    });
  }

  @override
  void didUpdateWidget(DynamicRiskBarChart oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Se cambia la categoria, resettiamo e facciamo risalire le barre
    if (oldWidget.categoryKey != widget.categoryKey) {
      _isLoaded = false;
      Future.delayed(const Duration(milliseconds: 50), () {
        if (mounted) setState(() => _isLoaded = true);
      });
    }
  }

  double _getCount(String categoryValue, String riskType) {
    try {
      return widget.rawData
          .firstWhere(
            (e) =>
                e[widget.categoryKey].toString() == categoryValue &&
                e['Risk Category'] == riskType,
          )['count']
          .toDouble();
    } catch (_) {
      return 0.0;
    }
  }

  double _calculateMaxY() {
    double max = 0;
    for (var e in widget.rawData) {
      if (e['count'] > max) max = e['count'].toDouble();
    }
    return max * 1.15; // Margine superiore
  }

  FlTitlesData _buildTitles(List<String> categories) {
    return FlTitlesData(
      show: true,
      // Gestione Asse X (Sotto)
      bottomTitles: AxisTitles(
        sideTitles: SideTitles(
          showTitles: true,
          reservedSize: 30,
          getTitlesWidget: (value, meta) {
            int index = value.toInt();
            // Evitiamo errori se l'indice è fuori range durante l'animazione
            if (index < 0 || index >= categories.length) {
              return const SizedBox();
            }

            return SideTitleWidget(
              space: 8, // Spazio tra la barra e il testo
              meta: meta,
              child: Text(
                _formatLabel(categories[index]),
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            );
          },
        ),
      ),
      // Nascondiamo gli altri assi per un look pulito "Glass"
      leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
      topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
      rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
    );
  }

  // Funzione opzionale per rendere le etichette più leggibili
  String _formatLabel(String label) {
    if (widget.categoryKey == "Decade") {
      return "$label-${int.parse(label) + 9}"; // trasforma "10" in "10-19"
    }
    return label;
  }

  @override
  Widget build(BuildContext context) {
    final categories = widget.rawData
        .map((e) => e[widget.categoryKey].toString())
        .toSet()
        .toList();
    categories.sort();

    return AspectRatio(
      aspectRatio: 2,
      child: BarChart(
        duration: const Duration(milliseconds: 800), // Durata dell'animazione
        curve: Curves.easeOutQuart,
        BarChartData(
          alignment: BarChartAlignment.spaceAround,
          maxY: _calculateMaxY(),
          titlesData: _buildTitles(categories),
          gridData: const FlGridData(show: false),
          borderData: FlBorderData(show: false),
          barGroups: List.generate(categories.length, (index) {
            String catName = categories[index];

            // Se _isLoaded è false, toY sarà 0 (le barre partono dal basso)
            double highCount = _isLoaded ? _getCount(catName, "High Risk") : 0;
            double lowCount = _isLoaded ? _getCount(catName, "Low Risk") : 0;

            return BarChartGroupData(
              x: index,
              barRods: [
                BarChartRodData(
                  toY: highCount,
                  color: Colors.redAccent,
                  width: 22,
                  borderRadius: const BorderRadius.vertical(
                    top: Radius.circular(4),
                  ),
                ),
                BarChartRodData(
                  toY: lowCount,
                  color: Colors.tealAccent,
                  width: 22,
                  borderRadius: const BorderRadius.vertical(
                    top: Radius.circular(4),
                  ),
                ),
              ],
            );
          }),
        ),
      ),
    );
  }
}
