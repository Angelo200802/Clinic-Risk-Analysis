import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

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
    return ScatterChart(
      ScatterChartData(
        scatterSpots: scatterData.map((point) {
          final isHighRisk = point['Risk Category'] == "High Risk";
          final x = (point[xKey] as num? ?? 0).toDouble();
          final y = (point[yKey] as num? ?? 0).toDouble();

          return ScatterSpot(
            x,
            y,
            dotPainter: FlDotCirclePainter(
              radius: 4,
              color: isHighRisk
                  ? highRiskColor.withOpacity(0.7)
                  : lowRiskColor.withOpacity(0.5),
              strokeWidth: 1,
              strokeColor: Colors.white,
            ),
          );
        }).toList(),
        titlesData: FlTitlesData(
          bottomTitles: AxisTitles(
            axisNameWidget: Text(
              xLabel,
              style: TextStyle(color: Colors.white70),
            ),
            sideTitles: SideTitles(showTitles: true, reservedSize: 30),
          ),
          leftTitles: AxisTitles(
            axisNameWidget: Text(
              yLabel,
              style: TextStyle(color: Colors.white70),
            ),
            sideTitles: SideTitles(showTitles: true, reservedSize: 40),
          ),
        ),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          getDrawingHorizontalLine: (value) => FlLine(color: Colors.white10),
        ),
        borderData: FlBorderData(show: false),
      ),
    );
  }
}
