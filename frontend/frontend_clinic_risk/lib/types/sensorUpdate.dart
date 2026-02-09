class SensorUpdate {
  final int patientId;
  final int heartRate;
  final int respiratoryRate;
  final String timestamp;
  final double bodyTemperature;
  final double oxygenSaturation;
  final double systolicBloodPressure;
  final double diastolicBloodPressure;
  final int age;
  final String gender;
  final double weight;
  final double height;
  final double derivedHrv;
  final double derivedPulsePressure;
  final double derivedBmi;
  final double derivedMap;
  final String prediction;

  SensorUpdate({
    required this.patientId,
    required this.heartRate,
    required this.respiratoryRate,
    required this.timestamp,
    required this.bodyTemperature,
    required this.oxygenSaturation,
    required this.systolicBloodPressure,
    required this.diastolicBloodPressure,
    required this.age,
    required this.gender,
    required this.weight,
    required this.height,
    required this.derivedHrv,
    required this.derivedPulsePressure,
    required this.derivedBmi,
    required this.derivedMap,
    required this.prediction,
  });

  factory SensorUpdate.fromJson(Map<String, dynamic> json) {
    return SensorUpdate(
      // Per gli int e String usiamo il solito null-check
      patientId: json['Patient ID'] ?? 0,
      heartRate: json['Heart Rate'] ?? 0,
      respiratoryRate: json['Respiratory Rate'] ?? 0,
      timestamp: json['Timestamp'] ?? '',
      age: json['Age'] ?? 0,
      gender: json['Gender'] ?? 'Unknown',
      prediction: json['Prediction'] ?? 'Unknown',

      // Per i DOUBLE: usiamo .toDouble() su un valore che forziamo a essere num
      // (num ?? 0).toDouble() accetta sia 138 che 138.0 e non crasha se è null
      bodyTemperature: (json['Body Temperature'] as num? ?? 0.0).toDouble(),
      oxygenSaturation: (json['Oxygen Saturation'] as num? ?? 0.0).toDouble(),
      systolicBloodPressure: (json['Systolic Blood Pressure'] as num? ?? 0.0)
          .toDouble(),
      diastolicBloodPressure: (json['Diastolic Blood Pressure'] as num? ?? 0.0)
          .toDouble(),
      weight: (json['Weight (kg)'] as num? ?? 0.0).toDouble(),
      height: (json['Height (m)'] as num? ?? 0.0).toDouble(),
      derivedHrv: (json['Derived_HRV'] as num? ?? 0.0).toDouble(),
      derivedPulsePressure: (json['Derived_Pulse_Pressure'] as num? ?? 0.0)
          .toDouble(),
      derivedBmi: (json['Derived_BMI'] as num? ?? 0.0).toDouble(),
      derivedMap: (json['Derived_MAP'] as num? ?? 0.0).toDouble(),
    );
  }
}
