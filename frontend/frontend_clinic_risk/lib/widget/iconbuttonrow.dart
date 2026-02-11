import 'package:flutter/material.dart';

const List<Map<String, dynamic>> chartAttributes = [
  {'label': 'Heart Rate', 'icon': Icons.favorite, 'color': Colors.redAccent},
  {'label': 'SpO2', 'icon': Icons.water_drop, 'color': Colors.blueAccent},
  {
    'label': 'Temperature',
    'icon': Icons.thermostat,
    'color': Colors.orangeAccent,
  },
  {'label': 'Respiratory Rate', 'icon': Icons.air, 'color': Colors.greenAccent},
];

class IconButtonRow extends StatelessWidget {
  final void Function(String) onPressed;
  final bool Function(String) isSelected;

  const IconButtonRow({
    super.key,
    required this.onPressed,
    required this.isSelected,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: chartAttributes.map((attr) {
          bool selected = isSelected(attr['label']);
          return GestureDetector(
            onTap: () => onPressed(attr['label']),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              margin: const EdgeInsets.only(right: 12, bottom: 16),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              decoration: BoxDecoration(
                color: selected
                    ? attr['color'].withOpacity(0.2)
                    : Colors.white.withOpacity(0.05),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: selected ? attr['color'] : Colors.transparent,
                  width: 1.5,
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    attr['icon'],
                    size: 16,
                    color: selected ? attr['color'] : Colors.white38,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    attr['label'],
                    style: TextStyle(
                      color: selected ? Colors.white : Colors.white38,
                      fontWeight: selected
                          ? FontWeight.bold
                          : FontWeight.normal,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}
