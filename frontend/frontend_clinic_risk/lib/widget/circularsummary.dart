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
              child: CircularProgressIndicator(
                value: value / 100, // Es: 0.98 per 98%
                strokeWidth: 6,
                backgroundColor: Colors.white10,
                valueColor: AlwaysStoppedAnimation<Color>(color),
              ),
            ),
            Text(
              value.toStringAsFixed(0),
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: Colors.white,
              ),
            ),
          ],
        ),
        SizedBox(height: 8),
        Text(title, style: TextStyle(fontSize: 10, color: Colors.white54)),
      ],
    );
  }
}

class CircularSummaryPanel extends StatelessWidget {
  final Trend trend;

  const CircularSummaryPanel({super.key, required this.trend});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          CircularSummary(
            title: "SpO₂",
            value: trend.avgSpo2,
            color: Colors.redAccent,
          ),
          CircularSummary(
            title: "Risk Ratio",
            value: trend.riskRatio * 100,
            color: Colors.orangeAccent,
          ),
        ],
      ),
    );
  }
}
