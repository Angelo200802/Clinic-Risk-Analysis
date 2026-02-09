import 'package:flutter/material.dart';
import 'package:frontend_clinic_risk/widget/classification_panel.dart';

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

  @override
  Widget build(BuildContext context) {
    // Logica di stato (personalizzala in base ai tuoi threshold)
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
            // Bordo neon se selezionata
            border: Border.all(
              color: widget.isSelected
                  ? Colors.blueAccent.withOpacity(0.8)
                  : Colors.white.withOpacity(0.05),
              width: 1.5,
            ),
            // Glow dinamico: rosso se critico, blu se selezionato
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
              children: [
                // Indicatore di stato verticale (Barra laterale)
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
                // Contenuto Card
                Padding(
                  padding: const EdgeInsets.fromLTRB(18, 14, 16, 14),
                  child: Row(
                    children: [
                      // Info Paziente
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
                                fontFamily:
                                    'monospace', // Garantisce stabilità visiva
                              ),
                            ),
                          ],
                        ),
                      ),
                      // Dato Vitale (BPM)
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

    return Container(
      color: Colors.black12,
      child: Row(
        children: [
          // COLONNA CRITICI
          Expanded(
            child: _buildTriageColumn(
              title: "High Risk",
              count: criticalPatients.length,
              color: Colors.redAccent,
              patients: criticalPatients,
            ),
          ),

          const VerticalDivider(width: 1, color: Colors.white10),

          // COLONNA STABILI
          Expanded(
            child: _buildTriageColumn(
              title: "Low Risk",
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
