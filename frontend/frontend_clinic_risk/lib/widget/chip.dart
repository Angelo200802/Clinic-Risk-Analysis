import 'package:flutter/material.dart';

class Chip extends StatelessWidget {
  final String label;
  final bool isActive;
  final bool isAlert;

  const Chip(
    this.label, {
    this.isActive = false,
    this.isAlert = false,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        // Se attivo usa il blu, se alert il rosso, altrimenti trasparente/grigio
        color: isActive
            ? Colors.blueAccent.withOpacity(0.1)
            : (isAlert
                  ? Colors.redAccent.withOpacity(0.1)
                  : Colors.white.withOpacity(0.05)),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: isActive
              ? Colors.blueAccent.withOpacity(0.5)
              : (isAlert ? Colors.redAccent.withOpacity(0.5) : Colors.white10),
          width: 1,
        ),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: isActive
              ? Colors.blueAccent
              : (isAlert ? Colors.redAccent : Colors.white38),
          fontSize: 10,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
