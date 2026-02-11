class Trend {
  final int patientId;
  final double avgHr;
  final int maxHr;
  final int minHr;
  final double avgRr;
  final int maxRr;
  final int minRr;
  final double avgSpo2;
  final double maxSpo2;
  final double minSpo2;
  final double avgTemp;
  final double maxTemp;
  final double minTemp;
  final double avgMap;
  final double maxMap;
  final double minMap;
  final double avgHrv;
  final double maxHrv;
  final double minHrv;
  final double stdHr;
  final int nSamples;
  final double hrDelta;
  final double mapDelta;
  final double spo2Delta;
  final double hrvDelta;
  final bool shockRisk;
  final bool respFailureRisk;
  final bool sepsisRisk;
  final bool hemoInstability;
  final int clinicalRiskScore;
  final String bmiClass;
  final double riskRatio;
  final String timestamp;
  final String start;
  final String end;

  Trend({
    required this.patientId,
    required this.avgHr,
    required this.maxHr,
    required this.minHr,
    required this.avgRr,
    required this.maxRr,
    required this.minRr,
    required this.avgSpo2,
    required this.maxSpo2,
    required this.minSpo2,
    required this.avgTemp,
    required this.maxTemp,
    required this.minTemp,
    required this.avgMap,
    required this.maxMap,
    required this.minMap,
    required this.avgHrv,
    required this.maxHrv,
    required this.minHrv,
    required this.stdHr,
    required this.nSamples,
    required this.hrDelta,
    required this.mapDelta,
    required this.spo2Delta,
    required this.hrvDelta,
    required this.shockRisk,
    required this.respFailureRisk,
    required this.sepsisRisk,
    required this.hemoInstability,
    required this.clinicalRiskScore,
    required this.start,
    required this.end,
    required this.timestamp,
    required this.bmiClass,
    required this.riskRatio,
  });

  double fromLabel(String label) {
    switch (label) {
      case "Heart Rate":
        return avgHr;
      case "Respiratory Rate":
        return avgRr;
      case "SpO2":
        return avgSpo2;
      case "Temperature":
        return avgTemp;
      default:
        throw Exception("Label non riconosciuto: $label");
    }
  }

  factory Trend.fromJson(Map<String, dynamic> json) {
    return Trend(
      patientId: json['Patient ID'] ?? 0,
      avgHr: (json['avg_hr'] as num? ?? 0.0).toDouble(),
      maxHr: json['max_hr'] ?? 0,
      minHr: json['min_hr'] ?? 0,
      avgRr: (json['avg_rr'] as num? ?? 0.0).toDouble(),
      maxRr: json['max_rr'] ?? 0,
      minRr: json['min_rr'] ?? 0,
      avgSpo2: (json['avg_spo2'] as num? ?? 0.0).toDouble(),
      maxSpo2: (json['max_spo2'] as num? ?? 0.0).toDouble(),
      minSpo2: (json['min_spo2'] as num? ?? 0.0).toDouble(),
      avgTemp: (json['avg_temp'] as num? ?? 0.0).toDouble(),
      maxTemp: (json['max_temp'] as num? ?? 0.0).toDouble(),
      minTemp: (json['min_temp'] as num? ?? 0.0).toDouble(),
      avgMap: (json['avg_map'] as num? ?? 0.0).toDouble(),
      maxMap: (json['max_map'] as num? ?? 0.0).toDouble(),
      minMap: (json['min_map'] as num? ?? 0.0).toDouble(),
      avgHrv: (json['avg_hrv'] as num? ?? 0.0).toDouble(),
      maxHrv: (json['max_hrv'] as num? ?? 0.0).toDouble(),
      minHrv: (json['min_hrv'] as num? ?? 0.0).toDouble(),
      stdHr: (json['std_hr'] as num? ?? 0.0).toDouble(),
      nSamples: json['n_samples'] ?? 0,
      hrDelta: (json['hr_delta'] as num? ?? 0.0).toDouble(),
      mapDelta: (json['map_delta'] as num? ?? 0.0).toDouble(),
      spo2Delta: (json['spo2_delta'] as num? ?? 0.0).toDouble(),
      hrvDelta: (json['hrv_delta'] as num? ?? 0.0).toDouble(),
      shockRisk: json['shock_risk'] == 1,
      respFailureRisk: json['resp_failure_risk'] == 1,
      sepsisRisk: json['sepsis_risk'] == 1,
      hemoInstability: json['hemo_instability'] == 1,
      clinicalRiskScore: json['clinical_risk_score'] ?? 0,
      start: json['start'] ?? '',
      end: json['end'] ?? '',
      timestamp: json['Timestamp'] ?? '',
      bmiClass: json['bmi_class'] ?? '',
      riskRatio: (json['risk_ratio'] as num? ?? 0.0).toDouble(),
    );
  }
}
