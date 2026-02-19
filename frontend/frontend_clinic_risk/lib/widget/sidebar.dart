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
          SidebarItem(
            icon: Icons.analytics_outlined,
            label: 'Analytics',
            isActive: selectedIndex == 1,
            isCollapsed: isCollapsed,
            onTap: () => onItemSelected(1),
          ),
          SidebarItem(
            icon: Icons.insights_outlined,
            label: 'Insights',
            isActive: selectedIndex == 2,
            isCollapsed: isCollapsed,
            onTap: () => onItemSelected(2),
          ),
          SidebarItem(
            icon: Icons.align_vertical_bottom_sharp,
            label: 'Evaluation',
            isActive: selectedIndex == 3,
            isCollapsed: isCollapsed,
            onTap: () => onItemSelected(3),
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
              "Health Dashboard",
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
  bool _isHovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: () {
          debugPrint("Tapped on: ${widget.label}");
          widget.onTap();
        },
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          // Riduciamo i margini orizzontali quando è contratto per centrare l'icona
          margin: EdgeInsets.symmetric(
            horizontal: widget.isCollapsed ? 10 : 15,
            vertical: 4,
          ),
          padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 12),
          decoration: BoxDecoration(
            color: widget.isActive
                ? Colors.greenAccent.withOpacity(0.15)
                : (_isHovered
                      ? Colors.white.withOpacity(0.05)
                      : Colors.transparent),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            // Centra l'icona quando la sidebar è contratta
            mainAxisAlignment: widget.isCollapsed
                ? MainAxisAlignment.center
                : MainAxisAlignment.start,
            children: [
              Icon(
                widget.icon,
                color: (widget.isActive || _isHovered)
                    ? Colors.greenAccent
                    : Colors.white54,
                size: 22,
              ),
              // Mostra lo spazio e il testo SOLO se non è contratto
              if (!widget.isCollapsed) ...[
                const SizedBox(width: 15),
                Flexible(
                  // Evita errori di overflow se il testo è lungo
                  child: Text(
                    widget.label,
                    overflow:
                        TextOverflow.ellipsis, // Taglia il testo se non ci sta
                    style: TextStyle(
                      color: (widget.isActive || _isHovered)
                          ? Colors.white
                          : Colors.white54,
                      fontWeight: widget.isActive
                          ? FontWeight.bold
                          : FontWeight.normal,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
