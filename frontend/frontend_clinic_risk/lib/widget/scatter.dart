import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class MetabolicEffortChart extends StatelessWidget {
  final List<dynamic> scatterData;

  const MetabolicEffortChart({super.key, required this.scatterData});

  @override
  Widget build(BuildContext context) {
    return ScatterChart(
      ScatterChartData(
        scatterSpots: scatterData.map((point) {
          final isHighRisk = point['Risk Category'] == "High Risk";
          final x = (point['ShockIndex'] as num? ?? 0).toDouble();
          final y = (point['PulsePressureIndex'] as num? ?? 0).toDouble();

          return ScatterSpot(
            x,
            y,
            dotPainter: FlDotCirclePainter(
              radius: isHighRisk ? 5 : 3,
              color: isHighRisk
                  ? Colors.orangeAccent
                  : Colors.blueAccent.withOpacity(0.5),
              strokeWidth: isHighRisk ? 1 : 0,
              strokeColor: Colors.white,
            ),
          );
        }).toList(),
        titlesData: FlTitlesData(
          bottomTitles: AxisTitles(
            axisNameWidget: Text(
              "Shock Index (HR / SBP)",
              style: TextStyle(color: Colors.white70),
            ),
            sideTitles: SideTitles(showTitles: true, reservedSize: 30),
          ),
          leftTitles: AxisTitles(
            axisNameWidget: Text(
              "Pulse Pressure Index (PP / HR)",
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
