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
    required this.lineColor,
    required this.label,
  });

  @override
  State<PatientTrendChart> createState() => _PatientTrendChartState();
}

class _PatientTrendChartState extends State<PatientTrendChart> {
  final double _viewWindowInMs = 3 * 60 * 1000;

  @override
  Widget build(BuildContext context) {
    // 1. Convertiamo e ORDINIAMO i dati per timestamp (fondamentale)
    List<FlSpot> allSpots =
        widget.history.map((e) {
          return FlSpot(
            DateTime.parse(e.timestamp).millisecondsSinceEpoch.toDouble(),
            e.fromLabel(widget.label),
          );
        }).toList()..sort(
          (a, b) => a.x.compareTo(b.x),
        ); // Ordiniamo in modo crescente per evitare i "nodi" nel grafico

    List<FlSpot> cleanSpots = [];
    for (var spot in allSpots) {
      if (cleanSpots.isEmpty || spot.x > cleanSpots.last.x) {
        cleanSpots.add(spot);
      } else {
        // Se il timestamp è uguale, sovrascriviamo l'ultimo valore Y
        // invece di aggiungere un punto sovrapposto
        cleanSpots[cleanSpots.length - 1] = spot;
      }
    }
    allSpots = cleanSpots;

    if (allSpots.isEmpty) {
      return const Center(child: Text("Nessun dato disponibile"));
    }

    // 2. FILTRIAMO i dati: passiamo al grafico SOLO quelli nel range visibile
    // Questo risolve i glitch grafici e migliora le performance
    double maxX = allSpots.last.x;
    double minX = maxX - _viewWindowInMs;

    List<FlSpot> visibleSpots = allSpots
        .where((s) => s.x >= (minX - 5000))
        .toList();

    // 3. Calcolo dinamico del range Y basato solo sui dati visibili
    // Se non ci sono abbastanza dati visibili, usiamo valori di default
    double minY = 0;
    double maxY = 100;

    if (visibleSpots.isNotEmpty) {
      minY = visibleSpots.map((s) => s.y).reduce((a, b) => a < b ? a : b);
      maxY = visibleSpots.map((s) => s.y).reduce((a, b) => a > b ? a : b);

      // Aggiungiamo un po' di padding verticale
      double padding = (maxY - minY).abs() * 0.1;
      minY -= (padding == 0 ? 10 : padding);
      maxY += (padding == 0 ? 10 : padding);
    }

    return LineChart(
      LineChartData(
        minX: minX,
        maxX: maxX,
        minY: minY,
        maxY: maxY,

        lineTouchData: LineTouchData(
          enabled: true,
          touchTooltipData: LineTouchTooltipData(
            // Colore di sfondo del tooltip (scuro e semi-trasparente)
            getTooltipColor: (touchedSpot) =>
                const Color(0xFF2C2C2C).withOpacity(0.9),
            tooltipBorderRadius: BorderRadius.circular(8),
            tooltipBorder: const BorderSide(color: Colors.white10),
            getTooltipItems: (List<LineBarSpot> touchedSpots) {
              return touchedSpots.map((LineBarSpot touchedSpot) {
                final DateTime date = DateTime.fromMillisecondsSinceEpoch(
                  touchedSpot.x.toInt(),
                );
                final String timeStr =
                    "${date.hour}:${date.minute.toString().padLeft(2, '0')}:${date.second.toString().padLeft(2, '0')}";

                return LineTooltipItem(
                  // Testo del Tooltip
                  '$timeStr\n',
                  const TextStyle(
                    color: Colors.white54,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                  children: [
                    TextSpan(
                      text:
                          '${touchedSpot.y.toStringAsFixed(1)} ${widget.label}',
                      style: TextStyle(
                        color: widget
                            .lineColor, // Usa lo stesso colore della linea
                        fontSize: 14,
                        fontWeight: FontWeight.w900,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                );
              }).toList();
            },
          ),
          // Mostra una linea verticale che segue il tocco
          getTouchedSpotIndicator:
              (LineChartBarData barData, List<int> spotIndexes) {
                return spotIndexes.map((index) {
                  return TouchedSpotIndicatorData(
                    FlLine(
                      color: widget.lineColor.withOpacity(0.3),
                      strokeWidth: 2,
                    ),
                    FlDotData(
                      show: true,
                      getDotPainter: (spot, percent, barData, index) =>
                          FlDotCirclePainter(
                            radius: 4,
                            color: Colors.white,
                            strokeWidth: 2,
                            strokeColor: widget.lineColor,
                          ),
                    ),
                  );
                }).toList();
              },
          handleBuiltInTouches:
              true, // Fondamentale per far funzionare il tocco
        ),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: true,
          drawHorizontalLine: true,
          // Opzionale: non disegnare la linea della griglia dove c'è già l'asse
          checkToShowHorizontalLine: (value) => value != minY,
          checkToShowVerticalLine: (value) => value != minX,
          getDrawingHorizontalLine: (value) =>
              FlLine(color: Colors.white.withOpacity(0.05), strokeWidth: 1),
          getDrawingVerticalLine: (value) =>
              FlLine(color: Colors.white.withOpacity(0.05), strokeWidth: 1),
        ),
        titlesData: FlTitlesData(
          show: true,
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 40,
              getTitlesWidget: (value, meta) {
                return SideTitleWidget(
                  space: 8,
                  meta: meta,
                  child: Text(
                    value.toInt().toString(),
                    style: const TextStyle(color: Colors.white54, fontSize: 10),
                  ),
                );
              },
            ),
          ),
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              interval: _viewWindowInMs / 3,
              getTitlesWidget: (value, meta) {
                final date = DateTime.fromMillisecondsSinceEpoch(value.toInt());
                // Mostriamo MM:SS per leggere bene lo scorrimento
                return SideTitleWidget(
                  meta: meta,
                  child: Text(
                    "${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}:${date.second.toString().padLeft(2, '0')}",
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
            spots: visibleSpots, // Passiamo solo i punti filtrati
            isCurved: true,
            curveSmoothness: 0.05,
            preventCurveOverShooting: true,
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
                  widget.lineColor.withOpacity(0.2),
                  widget.lineColor.withOpacity(0.0),
                ],
              ),
            ),
          ),
        ],
      ),
      // Disabilita l'animazione di default per evitare lo scatto del grafico
      duration: Duration.zero,
    );
  }
}
