import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'dart:math';

class ClinicScatterChart extends StatelessWidget {
  final List<dynamic> scatterData;
  final String xKey;
  final String yKey;
  final String xLabel;
  final String yLabel;
  final Color highRiskColor;
  final Color lowRiskColor;

  const ClinicScatterChart({
    super.key,
    required this.scatterData,
    required this.xKey,
    required this.yKey,
    required this.xLabel,
    required this.yLabel,
    required this.highRiskColor,
    required this.lowRiskColor,
  });

  @override
  Widget build(BuildContext context) {
    if (scatterData.isEmpty)
      return const Center(child: Text("Dati non disponibili"));

    // 1. Estrazione sicura dei valori
    final xValues = scatterData
        .map((e) => (e[xKey] as num? ?? 0).toDouble())
        .toList();
    final yValues = scatterData
        .map((e) => (e[yKey] as num? ?? 0).toDouble())
        .toList();

    double minX = xValues.reduce(min);
    double maxX = xValues.reduce(max);
    double minY = yValues.reduce(min);
    double maxY = yValues.reduce(max);

    // 2. Protezione contro range zero (il motivo del crash Infinity/NaN)
    if (minX == maxX) {
      minX -= 10;
      maxX += 10;
    }
    if (minY == maxY) {
      minY -= 10;
      maxY += 10;
    }

    // 3. Calcolo intervalli FISSI. Usiamo numeri interi per evitare arrotondamenti NaN
    // Importante: non lasciare che fl_chart decida l'interval
    double xInterval = ((maxX - minX) / 4).clamp(1.0, double.infinity);
    double yInterval = ((maxY - minY) / 4).clamp(1.0, double.infinity);

    return ScatterChart(
      ScatterChartData(
        minX: minX,
        maxX: maxX,
        minY: minY,
        maxY: maxY,
        scatterSpots: scatterData.map((point) {
          final isHighRisk = point['Risk Category'] == "High Risk";
          final x = (point[xKey] as num? ?? 0).toDouble();
          final y = (point[yKey] as num? ?? 0).toDouble();

          return ScatterSpot(
            x,
            y,
            dotPainter: FlDotCirclePainter(
              radius: isHighRisk ? 6 : 4,
              color: isHighRisk ? highRiskColor : lowRiskColor.withOpacity(0.6),
              strokeWidth: 1,
              strokeColor: Colors.white,
            ),
          );
        }).toList(),
        titlesData: FlTitlesData(
          show: true,
          bottomTitles: AxisTitles(
            axisNameWidget: Text(
              xLabel,
              style: const TextStyle(color: Colors.white70),
            ),
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              // FORZIAMO l'intervallo. Se non lo metti, la libreria usa
              // getEfficientInterval() che causa il crash NaN/Infinity
              interval: xInterval,
              getTitlesWidget: (value, meta) => Text(
                value.toInt().toString(),
                style: const TextStyle(color: Colors.white54, fontSize: 10),
              ),
            ),
          ),
          leftTitles: AxisTitles(
            axisNameWidget: Text(
              yLabel,
              style: const TextStyle(color: Colors.white70),
            ),
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 40,
              interval: yInterval, // FORZIAMO l'intervallo
              getTitlesWidget: (value, meta) => Text(
                value.toInt().toString(),
                style: const TextStyle(color: Colors.white54, fontSize: 10),
              ),
            ),
          ),
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
        ),
        gridData: FlGridData(
          show: true,
          horizontalInterval: yInterval,
          verticalInterval: xInterval,
          getDrawingHorizontalLine: (value) =>
              const FlLine(color: Colors.white10, strokeWidth: 1),
          getDrawingVerticalLine: (value) =>
              const FlLine(color: Colors.white10, strokeWidth: 1),
        ),
        borderData: FlBorderData(show: false),
      ),
    );
  }
}
