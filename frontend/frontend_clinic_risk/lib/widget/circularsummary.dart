import 'package:flutter/material.dart';
import 'package:frontend_clinic_risk/types/trend.dart';

class CircularSummary extends StatelessWidget {
  final String title;
  final double value; // Valore da 0 a 100
  final Color color;

  const CircularSummary({
    super.key,
    required this.title,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Stack(
          alignment: Alignment.center,
          children: [
            SizedBox(
              width: 60,
              height: 60,
              child: TweenAnimationBuilder<double>(
                // L'animazione va dal valore precedente al nuovo
                tween: Tween<double>(begin: 0, end: value / 100),
                duration: const Duration(milliseconds: 800),
                curve: Curves.easeOutCubic, // Animazione fluida e "organica"
                builder: (context, animatedValue, child) {
                  return CircularProgressIndicator(
                    value: animatedValue,
                    strokeWidth: 6,
                    backgroundColor: Colors.white10,
                    valueColor: AlwaysStoppedAnimation<Color>(color),
                    color: Colors.blueAccent,
                  );
                },
              ),
            ),
            // Anche il testo può beneficiare di un leggero effetto ombra per il "glow"
            Text(
              value.toStringAsFixed(0),
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: Colors.white,
                shadows: [Shadow(color: color.withOpacity(0.5), blurRadius: 8)],
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Text(
          title.toUpperCase(),
          style: const TextStyle(
            fontSize: 9,
            color: Colors.white54,
            letterSpacing: 1.1,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}

class CircularSummaryPanel extends StatelessWidget {
  final Trend trend;

  const CircularSummaryPanel({super.key, required this.trend});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      physics: const BouncingScrollPhysics(),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularSummary(
                title: "SpO₂",
                value: trend.avgSpo2,
                color: Colors.redAccent,
              ),
              const SizedBox(width: 20),
              CircularSummary(
                title: "Risk Ratio",
                value: trend.riskRatio * 100,
                color: Colors.orangeAccent,
              ),
              const SizedBox(width: 20),
              CircularSummary(
                title: "HRV",
                value: trend.avgHrv,
                color: Colors.blueAccent,
              ),
            ],
          ),
          const SizedBox(width: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Column(
                children: [
                  Text(
                    "BMI CLASS",
                    style: const TextStyle(
                      fontSize: 9,
                      color: Colors.white54,
                      letterSpacing: 1.1,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    trend.bmiClass,
                    style: const TextStyle(
                      fontSize: 14,
                      color: Colors.white,
                      letterSpacing: 1.1,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              const SizedBox(width: 40),
              Column(
                children: [
                  Text(
                    "Number of Samples",
                    style: const TextStyle(
                      fontSize: 9,
                      color: Colors.white54,
                      letterSpacing: 1.1,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    trend.nSamples.toString(),
                    style: const TextStyle(
                      fontSize: 14,
                      color: Colors.white,
                      letterSpacing: 1.1,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }
}
