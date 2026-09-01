import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class ClinicalRadarChart extends StatelessWidget {
  final List<dynamic> radarData;

  const ClinicalRadarChart({super.key, required this.radarData});

  // Definizione centralizzata di etichetta + chiave dati per ogni vertice,
  // così l'ordine resta sincronizzato tra assi e valori mostrati.
  static const List<Map<String, String>> _vertexDefs = [
    {'label': 'SI', 'key': 'Avg_ShockIndex'},
    {'label': 'MSI', 'key': 'Avg_ModifiedShockIndex'},
    {'label': 'AgeSI', 'key': 'Avg_AgeShockIndex_Norm'},
    {'label': 'DSI', 'key': 'Avg_DiastolicShockIndex'},
    {'label': 'PPI', 'key': 'Avg_PulsePressureIndex'},
  ];

  String _formatValue(dynamic data, String key) {
    if (data == null || data is! Map || data.isEmpty || data[key] == null) {
      return '-';
    }
    final v = data[key];
    if (v is num) return v.toDouble().toStringAsFixed(2);
    return v.toString();
  }

  @override
  Widget build(BuildContext context) {
    dynamic highRisk;
    try {
      highRisk = radarData.firstWhere(
        (element) => element["Risk Category"] == "High Risk",
      );
    } catch (e) {
      highRisk = {};
    }

    dynamic lowRisk;
    try {
      lowRisk = radarData.firstWhere(
        (element) => element["Risk Category"] == "Low Risk",
      );
    } catch (e) {
      lowRisk = {};
    }

    List<RadarDataSet> dataSets = [];
    if (highRisk.isNotEmpty) {
      dataSets.add(
        RadarDataSet(
          fillColor: Colors.redAccent.withOpacity(0.3),
          borderColor: Colors.redAccent,
          entryRadius: 3,
          dataEntries: _extractEntries(highRisk),
          borderWidth: 2,
        ),
      );
    }
    if (lowRisk.isNotEmpty) {
      dataSets.add(
        RadarDataSet(
          fillColor: Colors.greenAccent.withOpacity(0.3),
          borderColor: Colors.greenAccent,
          entryRadius: 3,
          dataEntries: _extractEntries(lowRisk),
          borderWidth: 2,
        ),
      );
    }

    return AspectRatio(
      aspectRatio: 1.3,
      child: RadarChart(
        RadarChartData(
          radarShape: RadarShape.polygon,
          radarBorderData: const BorderSide(color: Colors.white24, width: 1),
          gridBorderData: const BorderSide(color: Colors.white10, width: 1),
          tickBorderData: const BorderSide(color: Colors.white10, width: 1),
          ticksTextStyle: const TextStyle(color: Colors.white54, fontSize: 10),
          tickCount: 5,
          dataSets: dataSets,

          // Etichette sui vertici: nome + valore per High Risk / Low Risk
          getTitle: (index, angle) {
            if (index < 0 || index >= _vertexDefs.length) {
              return const RadarChartTitle(text: '');
            }
            final def = _vertexDefs[index];
            final label = def['label']!;
            final key = def['key']!;

            final hrVal = _formatValue(highRisk, key);
            final lrVal = _formatValue(lowRisk, key);

            return RadarChartTitle(
              text: '$label\nHR $hrVal · LR $lrVal',
              angle: angle,
            );
          },
          titleTextStyle: const TextStyle(color: Colors.white, fontSize: 11),
        ),
      ),
    );
  }
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
