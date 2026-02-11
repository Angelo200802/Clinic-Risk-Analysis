import 'dart:ui';

import 'package:flutter/material.dart';
import '../types/sensorUpdate.dart';

class PatientMiniCard extends StatefulWidget {
  final SensorUpdate patient; // Il tuo oggetto SensorUpdate
  final bool isSelected;
  final VoidCallback onTap;

  const PatientMiniCard({
    super.key,
    required this.patient,
    required this.isSelected,
    required this.onTap,
  });

  @override
  State<PatientMiniCard> createState() => _PatientMiniCardState();
}

class _PatientMiniCardState extends State<PatientMiniCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
      lowerBound: 0.95,
      upperBound: 1.05,
    );
  }

  @override
  void didUpdateWidget(PatientMiniCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Trigger dell'animazione al cambio dei BPM
    if (widget.patient.heartRate != oldWidget.patient.heartRate) {
      _pulseController.forward().then((_) => _pulseController.reverse());
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  void _showFullStats(BuildContext context, SensorUpdate patient) {
    showDialog(
      context: context,
      builder: (context) => BackdropFilter(
        filter: ImageFilter.blur(
          sigmaX: 5,
          sigmaY: 5,
        ), // Effetto vetro sullo sfondo
        child: AlertDialog(
          backgroundColor: const Color(0xFF1E1E1E),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          title: Row(
            children: [
              const Icon(Icons.analytics, color: Colors.blueAccent),
              const SizedBox(width: 10),
              Text(
                "Patient Summary: ${patient.patientId}",
                style: const TextStyle(color: Colors.white, fontSize: 18),
              ),
            ],
          ),
          content: SizedBox(
            width: 500,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildStatRow("Heart Rate", "${patient.heartRate} BPM"),
                _buildStatRow("Respiratory Rate", "${patient.respiratoryRate}"),
                _buildStatRow(
                  "SpO2",
                  "${patient.oxygenSaturation.toStringAsFixed(2)}%",
                ),
                _buildStatRow(
                  "Systolic Blood Pressure",
                  "${patient.systolicBloodPressure} mmHg",
                ),
                _buildStatRow(
                  "Diastolic Blood Pressure",
                  "${patient.diastolicBloodPressure} mmHg",
                ),
                _buildStatRow("Age", "${patient.age} y.o."),
                _buildStatRow("Gender", patient.gender),
                _buildStatRow(
                  "Body Temperature",
                  "${patient.bodyTemperature.toStringAsFixed(2)} °C",
                ),
                _buildStatRow(
                  "Height",
                  "${patient.height.toStringAsFixed(2)} m",
                ),
                _buildStatRow(
                  "Weight",
                  "${patient.weight.toStringAsFixed(2)} kg",
                ),
                _buildStatRow("BMI", "${patient.derivedBmi}"),
                _buildStatRow(
                  "MAP",
                  "${patient.derivedMap.toStringAsFixed(2)} mmHg",
                ),
                _buildStatRow(
                  "HRV",
                  "${patient.derivedHrv.toStringAsFixed(2)} ms",
                ),
                _buildStatRow(
                  "Pulse Pressure",
                  "${patient.derivedPulsePressure} mmHg",
                ),
                _buildStatRow("Risk Status", patient.prediction.toUpperCase()),
                _buildStatRow("Last Update", patient.timestamp),
                const SizedBox(height: 20),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text(
                "CLOSE",
                style: TextStyle(color: Colors.blueAccent),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.white.withOpacity(0.6))),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bool isCritical =
        widget.patient.prediction.toLowerCase() == "high risk";
    final Color statusColor = isCritical
        ? Colors.redAccent
        : Colors.greenAccent;

    return ScaleTransition(
      scale: _pulseController,
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 10),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                widget.isSelected
                    ? const Color(0xFF37474F)
                    : const Color(0xFF2C2C2C),
                widget.isSelected
                    ? const Color(0xFF263238)
                    : const Color(0xFF1A1A1A),
              ],
            ),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: widget.isSelected
                  ? Colors.blueAccent.withOpacity(0.8)
                  : Colors.white.withOpacity(0.05),
              width: 1.5,
            ),
            boxShadow: [
              if (isCritical)
                BoxShadow(
                  color: Colors.redAccent.withOpacity(0.2),
                  blurRadius: 12,
                  spreadRadius: 2,
                ),
              if (widget.isSelected)
                BoxShadow(
                  color: Colors.blueAccent.withOpacity(0.3),
                  blurRadius: 15,
                  spreadRadius: 1,
                ),
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Stack(
              // <--- Lo Stack comanda il posizionamento dei figli diretti
              children: [
                // 1. Indicatore laterale
                Positioned(
                  left: 0,
                  top: 0,
                  bottom: 0,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 500),
                    width: 6,
                    decoration: BoxDecoration(
                      color: statusColor,
                      boxShadow: [BoxShadow(color: statusColor, blurRadius: 6)],
                    ),
                  ),
                ),

                // 2. Contenuto Principale (ID e BPM)
                Padding(
                  padding: const EdgeInsets.fromLTRB(18, 14, 16, 14),
                  child: Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "PATIENT ID",
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.4),
                                fontSize: 9,
                                letterSpacing: 1.5,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              widget.patient.patientId.toString(),
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                fontFamily: 'monospace',
                              ),
                            ),
                          ],
                        ),
                      ),
                      // Aggiungiamo un po' di spazio per non far finire i BPM sotto l'occhio
                      const SizedBox(width: 35),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                Icons.favorite,
                                color: statusColor,
                                size: 14,
                              ),
                              const SizedBox(width: 4),
                              Text(
                                "${widget.patient.heartRate}",
                                style: TextStyle(
                                  color: statusColor,
                                  fontSize: 24,
                                  fontWeight: FontWeight.w900,
                                  fontFamily: 'monospace',
                                ),
                              ),
                            ],
                          ),
                          Text(
                            "BPM",
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.5),
                              fontSize: 10,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

                // 3. BOTTONE OCCHIO (Spostato qui come figlio diretto dello Stack)
                Positioned(
                  bottom: 2,
                  right:
                      42, // Regola questo valore per centrarlo tra ID e BPM o metterlo dove preferisci
                  child: Material(
                    color: Colors.transparent,
                    child: IconButton(
                      icon: Icon(
                        Icons.visibility_outlined,
                        size: 18,
                        color: Colors.white.withOpacity(0.4),
                      ),
                      hoverColor: Colors.blueAccent.withOpacity(0.1),
                      splashRadius: 20,
                      onPressed: () => _showFullStats(context, widget.patient),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class TriageMasterView extends StatelessWidget {
  final Map<int, SensorUpdate> allPatients;
  final int? selectedPatientId;
  final Function(SensorUpdate) onPatientSelected;

  const TriageMasterView({
    super.key,
    required this.allPatients,
    required this.selectedPatientId,
    required this.onPatientSelected,
  });

  @override
  Widget build(BuildContext context) {
    // Filtriamo i pazienti in base allo stato
    final criticalPatients = allPatients.values
        .where((p) => p.prediction.toLowerCase() == "high risk")
        .toList();
    final stablePatients = allPatients.values
        .where((p) => p.prediction.toLowerCase() != "high risk")
        .toList();

    double screenWidth = MediaQuery.of(context).size.width;
    double gap = (screenWidth * 0.02).clamp(8.0, 32.0);

    return Container(
      padding: EdgeInsets.all(gap), // Margine esterno variabile
      child: Row(
        children: [
          // BOX ALTO RISCHIO
          Expanded(
            child: _buildTriageColumn(
              title: "High Risk",
              count: criticalPatients.length,
              color: Colors.redAccent,
              patients: criticalPatients,
              glowColor: Colors.redAccent.withOpacity(0.05),
            ),
          ),

          // Distanziatore responsive
          SizedBox(width: gap),

          // BOX BASSO RISCHIO
          Expanded(
            child: _buildTriageColumn(
              title: "Low Risk",
              count: stablePatients.length,
              color: Colors.greenAccent,
              patients: stablePatients,
              glowColor: Colors.transparent, // Niente glow per chi sta bene
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState(Color color) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Icona con un leggero bagliore soffuso
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: color.withOpacity(0.05),
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.assignment_turned_in_outlined,
              color: color.withOpacity(0.4),
              size: 40,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            "NESSUN PAZIENTE",
            style: TextStyle(
              color: color.withOpacity(0.5),
              fontSize: 12,
              fontWeight: FontWeight.bold,
              letterSpacing: 1.1,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            "In questa categoria",
            style: TextStyle(
              color: Colors.white.withOpacity(0.2),
              fontSize: 11,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTriageColumn({
    required String title,
    required int count,
    required Color color,
    required List<SensorUpdate> patients,
    required Color glowColor,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(
          0xFF1A1A1A,
        ), // Sfondo box leggermente più chiaro del fondo app
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: color.withOpacity(
            0.2,
          ), // Bordo sottile colorato per richiamare lo stato
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(color: glowColor, blurRadius: 20, spreadRadius: 2),
        ],
      ),
      child: Column(
        children: [
          // Header del Box
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Icon(Icons.analytics_outlined, color: color, size: 20),
                    const SizedBox(width: 10),
                    Text(
                      title.toUpperCase(),
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.9),
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.2,
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
                // Badge numerico
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: color.withOpacity(0.3)),
                  ),
                  child: Text(
                    "$count",
                    style: TextStyle(
                      color: color,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
          ),

          const Divider(height: 1, color: Colors.white10),

          // Lista pazienti
          Expanded(
            child: patients.isEmpty
                ? _buildEmptyState(color)
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: patients.length,
                    itemBuilder: (context, index) => PatientMiniCard(
                      patient: patients[index],
                      isSelected:
                          patients[index].patientId == selectedPatientId,
                      onTap: () => onPatientSelected(patients[index]),
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}
