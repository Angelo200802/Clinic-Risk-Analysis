import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class ClinicalRadarChart extends StatelessWidget {
  final List<dynamic> radarData;

  const ClinicalRadarChart({super.key, required this.radarData});

  @override
  Widget build(BuildContext context) {
    // Estrazione dei dataset dal JSON
    final highRisk = radarData.firstWhere(
      (e) => e["Risk Category"] == "High Risk",
    );
    final lowRisk = radarData.firstWhere(
      (e) => e["Risk Category"] == "Low Risk",
    );

    return AspectRatio(
      aspectRatio: 1.3,
      child: RadarChart(
        RadarChartData(
          radarShape: RadarShape.polygon,

          radarBorderData: const BorderSide(color: Colors.white24, width: 1),
          gridBorderData: const BorderSide(color: Colors.white10, width: 1),
          tickBorderData: const BorderSide(color: Colors.white10, width: 1),
          ticksTextStyle: const TextStyle(color: Colors.white54, fontSize: 10),

          // Impostiamo il numero di cerchi della griglia
          tickCount: 5,

          dataSets: [
            // Dataset High Risk - Rosso
            RadarDataSet(
              fillColor: Colors.redAccent.withOpacity(0.3),
              borderColor: Colors.redAccent,
              entryRadius: 3,
              dataEntries: _extractEntries(highRisk),
              borderWidth: 2,
            ),
            // Dataset Low Risk - Blu
            RadarDataSet(
              fillColor: Colors.greenAccent.withOpacity(0.3),
              borderColor: Colors.greenAccent,
              entryRadius: 3,
              dataEntries: _extractEntries(lowRisk),
              borderWidth: 2,
            ),
          ],

          // Configurazione etichette sui vertici
          getTitle: (index, angle) {
            final double usedAngle = angle;
            switch (index) {
              case 0:
                return RadarChartTitle(text: 'SI', angle: usedAngle);
              case 1:
                return RadarChartTitle(text: 'MSI', angle: usedAngle);
              case 2:
                return RadarChartTitle(text: 'AgeSI', angle: usedAngle);
              case 3:
                return RadarChartTitle(text: 'DSI', angle: usedAngle);
              case 4:
                return RadarChartTitle(text: 'PPI', angle: usedAngle);
              default:
                return const RadarChartTitle(text: '');
            }
          },
          titleTextStyle: const TextStyle(color: Colors.white, fontSize: 12),
        ),
      ),
    );
  }

  List<RadarEntry> _extractEntries(Map<String, dynamic> data) {
    return [
      RadarEntry(value: data["Avg_ShockIndex"].toDouble()),
      RadarEntry(value: data["Avg_ModifiedShockIndex"].toDouble()),
      RadarEntry(value: data["Avg_AgeShockIndex_Norm"].toDouble()),
      RadarEntry(value: data["Avg_DiastolicShockIndex"].toDouble()),
      RadarEntry(value: data["Avg_PulsePressureIndex"].toDouble()),
    ];
  }
}
