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

    if (minX == maxX) {
      minX -= 10;
      maxX += 10;
    }

    double xInterval = (maxX - minX) / 4;
    double yInterval = (maxY - minY) / 4;
    if (xInterval <= 0) xInterval = 1.0;
    if (yInterval <= 0) yInterval = 1.0;

    return ScatterChart(
      ScatterChartData(
        minX: minX,
        maxX: maxX,
        minY: minY,
        maxY: maxY,
        // 1. Interazione al passaggio del mouse/tocco
        scatterTouchData: ScatterTouchData(
          enabled: true,
          touchTooltipData: ScatterTouchTooltipData(
            getTooltipItems: (ScatterSpot touchedSpot) {
              // 2. Risaliamo al punto originale tramite indice nella lista
              final index = scatterData.indexWhere((point) {
                final px = (point[xKey] as num? ?? 0).toDouble();
                final py = (point[yKey] as num? ?? 0).toDouble();
                return px == touchedSpot.x && py == touchedSpot.y;
              });

              if (index == -1) return null;

              final point = scatterData[index];
              final patientId = point['Patient ID']?.toString() ?? 'N/A';

              return ScatterTooltipItem(
                'ID: $patientId',
                textStyle: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              );
            },
          ),
        ),
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
              interval: xInterval,
              getTitlesWidget: (value, meta) => Text(
                value.toStringAsFixed(2),
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
              interval: yInterval,
              getTitlesWidget: (value, meta) => Text(
                value.toStringAsFixed(2),
                style: const TextStyle(color: Colors.white54, fontSize: 10),
              ),
            ),
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
