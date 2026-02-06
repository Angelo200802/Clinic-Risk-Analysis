import 'package:flutter/material.dart';
import 'threedots.dart';

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

class LiveClassificationPane extends StatefulWidget {
  final SensorUpdate sensorUpdate;
  final bool isConnected;

  const LiveClassificationPane({
    super.key,
    required this.sensorUpdate,
    required this.isConnected,
  });

  @override
  State<LiveClassificationPane> createState() => _LiveClassificationPaneState();
}

class _LiveClassificationPaneState extends State<LiveClassificationPane> {
  int _animState = 0;

  Color _getRiskColor() {
    switch (widget.sensorUpdate.prediction.toLowerCase()) {
      case 'low risk':
        return const Color.fromARGB(255, 36, 236, 42);
      case 'high risk':
        return const Color.fromARGB(255, 169, 17, 6);
      default:
        return Colors.grey;
    }
  }

  @override
  void didUpdateWidget(LiveClassificationPane oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.sensorUpdate != null &&
        widget.sensorUpdate != oldWidget.sensorUpdate) {
      _runSequence();
    }
  }

  void _runSequence() async {
    setState(() => _animState = 1);
    await Future.delayed(const Duration(milliseconds: 800));
    if (mounted) setState(() => _animState = 2);
    await Future.delayed(const Duration(seconds: 4));
    if (mounted) setState(() => _animState = 0);
  }

  @override
  Widget build(BuildContext context) {
    Color statusColor = _getRiskColor();

    // 1. Usiamo LayoutBuilder per rendere il contenuto relativo alle dimensioni del box
    return LayoutBuilder(
      builder: (context, constraints) {
        double maxWidth = constraints.maxWidth;
        // Calcoliamo i font in base alla larghezza del contenitore
        double bigFontSize = maxWidth * 0.05; // 10% della larghezza
        double smallFontSize = maxWidth * 0.05; // 5% della larghezza

        return AspectRatio(
          aspectRatio: 1.8, // Mantiene una forma armoniosa (simile a 16:9)
          child: Container(
            decoration: BoxDecoration(
              color: const Color(0xFF151515),
              borderRadius: BorderRadius.circular(
                maxWidth * 0.08,
              ), // Bordi arrotondati relativi
              border: Border.all(
                color: _animState == 2
                    ? statusColor.withOpacity(0.5)
                    : Colors.white10,
                width: 2,
              ),
            ),
            child: widget.isConnected
                ? ClipRRect(
                    borderRadius: BorderRadius.circular(maxWidth * 0.08),
                    child: Stack(
                      alignment: Alignment.center,
                      children: [
                        if (_animState == 0) const Threedots(),

                        if (_animState > 0)
                          AnimatedAlign(
                            duration: const Duration(milliseconds: 600),
                            curve: Curves.easeOutBack,
                            // L'ID sale di una percentuale fissa rispetto al centro
                            alignment: _animState == 1
                                ? Alignment.center
                                : const Alignment(0, -0.6),
                            child: AnimatedDefaultTextStyle(
                              duration: const Duration(milliseconds: 600),
                              style: TextStyle(
                                fontSize: _animState == 1
                                    ? bigFontSize
                                    : smallFontSize,
                                fontWeight: FontWeight.bold,
                                color: _animState == 1
                                    ? Colors.white
                                    : Colors.white70,
                              ),
                              child: Text(
                                "PATIENT ID: ${widget.sensorUpdate.patientId}",
                              ),
                            ),
                          ),

                        if (_animState == 2)
                          AnimatedAlign(
                            duration: const Duration(milliseconds: 400),
                            alignment: const Alignment(
                              0,
                              0.4,
                            ), // Label posizionata nel quadrante inferiore
                            child: TweenAnimationBuilder<double>(
                              tween: Tween(begin: 0.0, end: 1.0),
                              duration: const Duration(milliseconds: 400),
                              builder: (context, value, child) {
                                return Transform.scale(
                                  scale: value,
                                  child: Opacity(opacity: value, child: child),
                                );
                              },
                              child: Container(
                                padding: EdgeInsets.symmetric(
                                  horizontal: maxWidth * 0.08,
                                  vertical: maxWidth * 0.03,
                                ),
                                decoration: BoxDecoration(
                                  color: statusColor,
                                  borderRadius: BorderRadius.circular(
                                    maxWidth * 0.04,
                                  ),
                                  boxShadow: [
                                    BoxShadow(
                                      color: statusColor.withOpacity(0.3),
                                      blurRadius: 10,
                                    ),
                                  ],
                                ),
                                child: Text(
                                  widget.sensorUpdate.prediction.toUpperCase(),
                                  style: TextStyle(
                                    color: Colors.black,
                                    fontSize:
                                        maxWidth *
                                        0.06, // Testo label responsive
                                  ),
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  )
                : const Center(
                    child: Text(
                      "Unable to connect to data stream",
                      style: TextStyle(color: Colors.white54, fontSize: 18),
                    ),
                  ),
          ),
        );
      },
    );
  }
}
