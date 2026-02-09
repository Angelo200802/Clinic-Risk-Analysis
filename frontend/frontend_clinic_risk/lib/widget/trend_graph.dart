import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../types/trend.dart';

class PatientTrendChart extends StatefulWidget {
  final List<Trend> history;
  final Color lineColor;
  final String label;

  const PatientTrendChart({
    super.key,
    required this.history,
    this.lineColor = Colors.blueAccent,
    required this.label,
  });

  @override
  State<PatientTrendChart> createState() => _PatientTrendChartState();
}

class _PatientTrendChartState extends State<PatientTrendChart> {
  // Definiamo quanto tempo vogliamo visualizzare (es. gli ultimi 5 minuti)
  final double _viewWindowInMs = 5 * 60 * 1000;

  @override
  Widget build(BuildContext context) {
    List<FlSpot> spots = widget.history.map((e) {
      return FlSpot(
        DateTime.parse(e.start).millisecondsSinceEpoch.toDouble(),
        e.fromLabel(widget.label), // Convertiamo il valore in double
      );
    }).toList();

    if (spots.isEmpty) {
      return const Center(child: Text("Nessun dato disponibile"));
    }

    double maxX = spots.last.x;
    double minX = maxX - _viewWindowInMs;

    return LineChart(
      LineChartData(
        minX: minX,
        maxX: maxX,
        minY: spots.map((s) => s.y).reduce((a, b) => a < b ? a : b) * 0.9,
        maxY: spots.map((s) => s.y).reduce((a, b) => a > b ? a : b) * 1.1,

        lineTouchData: const LineTouchData(enabled: false),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: true,
          getDrawingHorizontalLine: (value) =>
              const FlLine(color: Colors.white10, strokeWidth: 1),
          getDrawingVerticalLine: (value) =>
              const FlLine(color: Colors.white10, strokeWidth: 1),
        ),
        titlesData: FlTitlesData(
          show: true,
          // Asse Y (Sinistro)
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize:
                  40, // <--- Aumenta questo valore se i numeri sono tagliati
              interval:
                  20, // Ogni quanto mostrare un numero (es. 60, 80, 100...)
              getTitlesWidget: (value, meta) {
                return SideTitleWidget(
                  space: 8, // Spazio tra il numero e il grafico
                  meta: meta,
                  child: Text(
                    value.toInt().toString(),
                    style: const TextStyle(
                      color: Colors.white54,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'monospace',
                    ),
                  ),
                );
              },
            ),
          ),
          // Disabilita gli altri assi che non servono
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          // Asse X (Basso)
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              getTitlesWidget: (value, meta) {
                // ... tua logica precedente per il tempo ...
                return SideTitleWidget(
                  meta: meta,
                  child: Text(
                    "${DateTime.fromMillisecondsSinceEpoch(value.toInt()).second}s",
                    style: const TextStyle(color: Colors.white24, fontSize: 10),
                  ),
                );
              },
            ),
          ),
        ),
        borderData: FlBorderData(show: false),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            curveSmoothness: 0.3,
            color: widget.lineColor,
            barWidth: 2,
            isStrokeCapRound: true,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  widget.lineColor.withOpacity(0.3),
                  widget.lineColor.withOpacity(0.0),
                ],
              ),
            ),
          ),
        ],
      ),
      duration: const Duration(
        milliseconds: 0,
      ), // Settare a 0 per real-time estremo
    );
  }
}
