import 'package:flutter/material.dart';

class SidebarComponent extends StatelessWidget {
  final int selectedIndex;
  final Function(int) onItemSelected;
  final bool isCollapsed;

  const SidebarComponent({
    super.key,
    required this.selectedIndex,
    required this.onItemSelected,
    required this.isCollapsed,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: isCollapsed ? 80 : 260,
      color: const Color(0xFF151515),
      child: Column(
        children: [
          _buildLogo(isCollapsed),
          const SizedBox(height: 20),
          SidebarItem(
            icon: Icons.sensors_off_rounded,
            label: 'Live Stream',
            isActive: selectedIndex == 0,
            isCollapsed: isCollapsed,
            onTap: () => onItemSelected(0),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget _buildLogo(bool isCollapsed) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 30),
      child: isCollapsed
          ? const Icon(Icons.bolt, color: Colors.greenAccent, size: 30)
          : const Text(
              "Menù",
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
    );
  }
}

class SidebarItem extends StatefulWidget {
  final IconData icon;
  final String label;
  final bool isActive;
  final bool isCollapsed;
  final VoidCallback onTap;

  const SidebarItem({
    super.key,
    required this.icon,
    required this.label,
    required this.isActive,
    required this.isCollapsed,
    required this.onTap,
  });

  @override
  State<SidebarItem> createState() => _SidebarItemState();
}

class _SidebarItemState extends State<SidebarItem> {
  bool _isHovered = false; // Traccia se il mouse è sopra l'elemento

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      cursor: SystemMouseCursors.click, // Cambia il cursore in una manina
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          margin: const EdgeInsets.symmetric(horizontal: 15, vertical: 4),
          padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 12),
          decoration: BoxDecoration(
            // Il colore cambia se è attivo OPPURE se il mouse è sopra
            color: widget.isActive
                ? Colors.greenAccent.withOpacity(0.15)
                : (_isHovered
                      ? Colors.white.withOpacity(0.05)
                      : Colors.transparent),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            children: [
              // L'icona si illumina se hoverata
              Icon(
                widget.icon,
                color: (widget.isActive || _isHovered)
                    ? Colors.greenAccent
                    : Colors.white54,
                size: 22,
              ),
              const SizedBox(width: 15),
              Text(
                widget.label,
                style: TextStyle(
                  color: (widget.isActive || _isHovered)
                      ? Colors.white
                      : Colors.white54,
                  fontWeight: widget.isActive
                      ? FontWeight.bold
                      : FontWeight.normal,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
