class Trend {
  final int patientId;
  final double avgHr;
  final double avgRr;
  final double avgSpo2;
  final double avgTemp;
  final double avgMap;
  final double avgHrv;
  final double stdHr;
  final double hrPct;
  final double rrPct;
  final double spo2Pct;
  final double ppPct;
  final double mapPct;
  final int nSamples;
  final String bmiClass;
  final double riskRatio;
  final String timestamp;
  final String start;
  final String end;

  Trend({
    required this.patientId,
    required this.avgHr,
    required this.avgRr,
    required this.avgSpo2,
    required this.avgTemp,
    required this.avgMap,
    required this.avgHrv,
    required this.stdHr,
    required this.nSamples,
    required this.start,
    required this.end,
    required this.timestamp,
    required this.bmiClass,
    required this.riskRatio,
    required this.hrPct,
    required this.rrPct,
    required this.spo2Pct,
    required this.ppPct,
    required this.mapPct,
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
      avgRr: (json['avg_rr'] as num? ?? 0.0).toDouble(),
      avgSpo2: (json['avg_spo2'] as num? ?? 0.0).toDouble(),
      avgTemp: (json['avg_temp'] as num? ?? 0.0).toDouble(),
      avgMap: (json['avg_map'] as num? ?? 0.0).toDouble(),
      avgHrv: (json['avg_hrv'] as num? ?? 0.0).toDouble(),
      stdHr: (json['std_hr'] as num? ?? 0.0).toDouble(),
      nSamples: json['n_samples'] ?? 0,
      hrPct: (json['hr_pct'] as num? ?? 0.0).toDouble(),
      rrPct: (json['rr_pct'] as num? ?? 0.0).toDouble(),
      spo2Pct: (json['spo2_pct'] as num? ?? 0.0).toDouble(),
      ppPct: (json['pp_pct'] as num? ?? 0.0).toDouble(),
      mapPct: (json['map_pct'] as num? ?? 0.0).toDouble(),
      start: json['start'] ?? '',
      end: json['end'] ?? '',
      timestamp: json['Timestamp'] ?? '',
      bmiClass: json['bmi_class'] ?? '',
      riskRatio: (json['risk_ratio'] as num? ?? 0.0).toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'Patient ID': patientId,
      'avg_hr': avgHr,
      'avg_rr': avgRr,
      'avg_spo2': avgSpo2,
      'avg_temp': avgTemp,
      'avg_map': avgMap,
      'avg_hrv': avgHrv,
      'std_hr': stdHr,
      'n_samples': nSamples,
      'hr_pct': hrPct,
      'rr_pct': rrPct,
      'spo2_pct': spo2Pct,
      'pp_pct': ppPct,
      'map_pct': mapPct,
      'start': start,
      'end': end,
      'Timestamp': timestamp,
      'bmi_class': bmiClass,
      'risk_ratio': riskRatio,
    };
  }
}
