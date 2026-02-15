import 'package:flutter/material.dart';

class StatRangeVisualizer extends StatelessWidget {
  final String title;
  final num min;
  final num max;
  final num mean;
  final num stdDev;
  final String unit;
  final int totalSamples;

  const StatRangeVisualizer({
    super.key,
    required this.title,
    required this.min,
    required this.max,
    required this.mean,
    required this.stdDev,
    this.unit = "",
    required this.totalSamples,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        children: [
          Text(
            "Statistics: $title",
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 25),

          // Row per i due indicatori circolari superiori
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildCircularIndicator(
                "Media",
                mean.toStringAsFixed(1),
                Icons.analytics,
                Colors.cyanAccent,
              ),
              _buildCircularIndicator(
                "Dev. Std",
                "±${stdDev.toStringAsFixed(1)}",
                Icons.architecture,
                Colors.purpleAccent,
              ),
            ],
          ),

          const SizedBox(height: 40),
          const Text(
            "Distribuzione Valori",
            style: TextStyle(color: Colors.white70, fontSize: 14),
          ),
          const SizedBox(height: 15),

          // La barra di distribuzione custom
          _buildDistributionBar(),

          const SizedBox(height: 10),
          // Etichette Min, Media e Max
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _buildLabel("Min", min),
              _buildLabel("Media", mean),
              _buildLabel("Max", max),
            ],
          ),

          const SizedBox(height: 30),
          Text(
            "Totale campioni analizzati: ${totalSamples.toString().replaceAllMapped(RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'), (Match m) => '${m[1]}.')}",
            style: const TextStyle(color: Colors.white38, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildCircularIndicator(
    String label,
    String value,
    IconData icon,
    Color color,
  ) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: color.withOpacity(0.5), width: 2),
            boxShadow: [
              BoxShadow(
                color: color.withOpacity(0.1),
                blurRadius: 10,
                spreadRadius: 2,
              ),
            ],
          ),
          child: Icon(icon, color: color, size: 28),
        ),
        const SizedBox(height: 10),
        Text(
          label,
          style: const TextStyle(color: Colors.white54, fontSize: 12),
        ),
        Text(
          "$value $unit",
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _buildDistributionBar() {
    return LayoutBuilder(
      builder: (context, constraints) {
        double width = constraints.maxWidth;
        // Calcolo posizioni relative (0.0 a 1.0)
        num range = max - min;
        double meanPos = (mean - min) / range;
        double stdDevWidth = (stdDev / range) * width;
        double meanX = meanPos * width;

        return Stack(
          alignment: Alignment.centerLeft,
          children: [
            // Background Bar
            Container(
              height: 12,
              width: width,
              decoration: BoxDecoration(
                color: Colors.white10,
                borderRadius: BorderRadius.circular(6),
              ),
            ),
            // Area Deviazione Standard (Ombreggiatura attorno alla media)
            Positioned(
              left: (meanX - stdDevWidth).clamp(0, width),
              child: Container(
                height: 24,
                width: (stdDevWidth * 2).clamp(
                  0,
                  width - (meanX - stdDevWidth),
                ),
                decoration: BoxDecoration(
                  color: Colors.cyanAccent.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
            ),
            // Linea della Media (Il "Tick" verticale)
            Positioned(
              left: meanX - 2,
              child: Container(
                height: 30,
                width: 4,
                decoration: BoxDecoration(
                  color: Colors.cyanAccent,
                  borderRadius: BorderRadius.circular(2),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.cyanAccent.withOpacity(0.5),
                      blurRadius: 8,
                    ),
                  ],
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildLabel(String title, num val) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Text(
          title,
          style: const TextStyle(color: Colors.white38, fontSize: 10),
        ),
        Text(
          "${val.toStringAsFixed(1)} $unit",
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}
