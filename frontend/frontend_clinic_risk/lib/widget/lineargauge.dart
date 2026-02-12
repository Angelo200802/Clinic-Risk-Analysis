import 'package:flutter/material.dart';
import 'package:frontend_clinic_risk/types/trend.dart';

class LinearGauge extends StatelessWidget {
  final double value;
  final Function(double) _getColor;

  const LinearGauge({
    super.key,
    required this.value,
    required Function(double) getColor,
  }) : _getColor = getColor;

  @override
  Widget build(BuildContext context) {
    // Proporzione del valore tra 34°C e 42°C (0.0 a 1.0)
    double percent = ((value - 34) / (42 - 34)).clamp(0.0, 1.0);
    const double gaugeHeight = 120.0;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          "${value.toStringAsFixed(1)}°",
          style: TextStyle(
            color: _getColor(value),
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
        const SizedBox(height: 8),
        SizedBox(
          width:
              30, // Larghezza leggermente maggiore per ospitare la sbarretta sporgente
          height: gaugeHeight,
          child: Stack(
            alignment: Alignment.bottomCenter,
            children: [
              // 1. LA COLONNA (Sfondo statico col gradiente)
              Container(
                width: 6,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(3),
                  gradient: const LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      Colors.blue,
                      Colors.green,
                      Colors.orange,
                      Colors.red,
                    ],
                    stops: [0.1, 0.4, 0.7, 0.9],
                  ),
                ),
              ),

              // 2. LA SBARRETTA ANIMATA (Il cursore di livello)
              AnimatedAlign(
                duration: const Duration(
                  milliseconds: 1000,
                ), // Più lenta per un look "analogico"
                curve: Curves.easeInOutCubic, // Partenza e arrivo morbidi
                // Alignment(0, 1) è la base (34°), Alignment(0, -1) è la cima (42°)
                alignment: Alignment(0, 1 - (percent * 2)),
                child: Container(
                  width: 24, // Più larga della colonna per "abbracciarla"
                  height: 3, // Spessore della sbarretta
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(2),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.4),
                        blurRadius: 2,
                        offset: const Offset(0, 1),
                      ),
                      // Effetto "Glow" che segue la sbarretta
                      BoxShadow(
                        color: _getColor(value).withOpacity(0.5),
                        blurRadius: 8,
                        spreadRadius: 1,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 6),
        const Text(
          "TEMP",
          style: TextStyle(
            color: Colors.white24,
            fontSize: 8,
            letterSpacing: 1,
          ),
        ),
      ],
    );
  }
}

class VerticalBulletChart extends StatelessWidget {
  final double value; // Valore attuale del MAP
  final double target; // Valore target (es. 90)

  const VerticalBulletChart({
    super.key,
    required this.value,
    required this.target,
  });

  @override
  Widget build(BuildContext context) {
    // Range tipici MAP: 40 (min) - 140 (max)
    double percentVal = ((value - 40) / (140 - 40)).clamp(0.0, 1.0);
    double percentTarget = ((target - 40) / (140 - 40)).clamp(0.0, 1.0);

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          value.toStringAsFixed(0),
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
        const SizedBox(height: 8),
        SizedBox(
          width: 30,
          height: 120,
          child: Stack(
            alignment: Alignment.bottomCenter,
            children: [
              // 1. Sfondo a zone (Range qualitativi)
              Container(
                width: 12,
                decoration: BoxDecoration(
                  color: Colors.white10,
                  borderRadius: BorderRadius.circular(2),
                ),
                child: Column(
                  children: [
                    Expanded(
                      flex: 40,
                      child: Container(color: Colors.red.withOpacity(0.1)),
                    ), // Alto
                    Expanded(
                      flex: 30,
                      child: Container(color: Colors.green.withOpacity(0.1)),
                    ), // Normale
                    Expanded(
                      flex: 30,
                      child: Container(color: Colors.blue.withOpacity(0.1)),
                    ), // Basso
                  ],
                ),
              ),
              // 2. Barra del valore attuale (più stretta dello sfondo)
              AnimatedContainer(
                duration: const Duration(milliseconds: 800),
                width: 6,
                height: 120 * percentVal,
                decoration: BoxDecoration(
                  color: _getMAPColor(value),
                  borderRadius: BorderRadius.circular(2),
                  boxShadow: [
                    BoxShadow(
                      color: _getMAPColor(value).withOpacity(0.5),
                      blurRadius: 4,
                    ),
                  ],
                ),
              ),
              // 3. Indicatore Target (Marker orizzontale)
              AnimatedPositioned(
                duration: const Duration(milliseconds: 800),
                bottom: 120 * percentTarget,
                child: Container(width: 20, height: 2, color: Colors.white),
              ),
            ],
          ),
        ),
        const SizedBox(height: 4),
        const Text("MAP", style: TextStyle(color: Colors.white24, fontSize: 8)),
      ],
    );
  }

  Color _getMAPColor(double v) {
    if (v < 65) return Colors.blueAccent; // Ipotensione
    if (v > 110) return Colors.redAccent; // Ipertensione
    return Colors.greenAccent; // Perfetto
  }
}

Widget getVertical(Trend trend, Function(double) getColor) {
  // 2° Quadrante: Vertical Vitals Panel
  return Expanded(
    child: Container(
      padding: const EdgeInsets.all(20.0),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A).withOpacity(0.5),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            "VITAL COLUMNS",
            style: TextStyle(
              color: Colors.white54,
              fontSize: 10,
              letterSpacing: 1.5,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 25),
          Row(
            mainAxisAlignment:
                MainAxisAlignment.center, // Centra le colonne nel quadrante
            crossAxisAlignment:
                CrossAxisAlignment.end, // Allinea le basi dei gauge
            children: [
              // Termometro Verticale
              LinearGauge(value: trend.avgTemp, getColor: getColor),
              const SizedBox(width: 40), // Spazio generoso tra i due
              // Bullet Chart per il MAP
              VerticalBulletChart(value: trend.avgMap, target: 90.0),
            ],
          ),
        ],
      ),
    ),
  );
}
