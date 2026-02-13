import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class RocCurveChart extends StatelessWidget {
  final List<FlSpot> points;

  const RocCurveChart({super.key, required this.points});

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1.5,
      child: LineChart(
        LineChartData(
          gridData: FlGridData(
            show: true,
            drawVerticalLine: true,
            getDrawingHorizontalLine: (value) =>
                FlLine(color: Colors.white10, strokeWidth: 1),
            getDrawingVerticalLine: (value) =>
                FlLine(color: Colors.white10, strokeWidth: 1),
          ),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(showTitles: true, reservedSize: 40),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(showTitles: true, reservedSize: 30),
            ),
            topTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
            rightTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
          ),
          borderData: FlBorderData(
            show: true,
            border: Border.all(color: Colors.white24),
          ),
          minX: 0,
          maxX: 1,
          minY: 0,
          maxY: 1,
          lineBarsData: [
            // Linea del Modello
            LineChartBarData(
              spots: points,
              isCurved: true,
              color: Colors.blueAccent,
              barWidth: 3,
              dotData: const FlDotData(show: false),
              belowBarData: BarAreaData(
                show: true,
                color: Colors.blueAccent.withOpacity(0.1),
              ),
            ),
            // Linea di Base (Casuale)
            LineChartBarData(
              spots: const [FlSpot(0, 0), FlSpot(1, 1)],
              isCurved: false,
              color: Colors.red.withOpacity(0.5),
              dashArray: [5, 5],
              barWidth: 2,
              dotData: const FlDotData(show: false),
            ),
          ],
        ),
      ),
    );
  }
}
