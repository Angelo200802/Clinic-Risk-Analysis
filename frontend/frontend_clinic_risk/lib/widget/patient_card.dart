import 'package:flutter/material.dart';
import 'package:frontend_clinic_risk/widget/classification_panel.dart';

class PatientMiniCard extends StatefulWidget {
  final SensorUpdate patient; // Oggetto che contiene i dati
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
      duration: const Duration(milliseconds: 300),
      lowerBound: 0.9,
      upperBound: 1.1,
    );
  }

  // Eseguiamo l'animazione ogni volta che il battito cambia
  @override
  void didUpdateWidget(PatientMiniCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.patient != oldWidget.patient) {
      _pulseController.forward().then((_) => _pulseController.reverse());
    }
  }

  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _pulseController,
      child: GestureDetector(
        onTap: widget.onTap,
        child: Container(
          // ... resto del design della card ...
          child: Text(
            "${widget.patient.heartRate} Heart Rate",
          ), // Esempio di dato da visualizzare
        ),
      ),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }
}

class TriageMasterView extends StatelessWidget {
  final List<SensorUpdate> allPatients;
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
    final criticalPatients = allPatients
        .where((SensorUpdate p) => p.prediction.toLowerCase() == "high risk")
        .toList();
    final stablePatients = allPatients
        .where((SensorUpdate p) => p.prediction.toLowerCase() != "high risk")
        .toList();

    return Container(
      color: Colors.black12,
      child: Row(
        children: [
          // COLONNA CRITICI
          Expanded(
            child: _buildTriageColumn(
              title: "CRITICI",
              count: criticalPatients.length,
              color: Colors.redAccent,
              patients: criticalPatients,
            ),
          ),

          const VerticalDivider(width: 1, color: Colors.white10),

          // COLONNA STABILI
          Expanded(
            child: _buildTriageColumn(
              title: "STABILI",
              count: stablePatients.length,
              color: Colors.greenAccent,
              patients: stablePatients,
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
  }) {
    return Column(
      children: [
        // Header della colonna
        Padding(
          padding: const EdgeInsets.all(12.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                title,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  "$count",
                  style: TextStyle(
                    color: color,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
        // Lista effettiva
        Expanded(
          child: ListView.builder(
            itemCount: patients.length,
            itemBuilder: (context, index) {
              final patient = patients[index];
              return PatientMiniCard(
                patient: patient,
                isSelected: patient.patientId == selectedPatientId,
                onTap: () => onPatientSelected(patient),
              );
            },
          ),
        ),
      ],
    );
  }
}
