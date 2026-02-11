import 'package:flutter/material.dart';
import 'threedots.dart';
import '../types/sensorUpdate.dart';

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
        return Colors.greenAccent;
      case 'high risk':
        return Colors.redAccent;
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
