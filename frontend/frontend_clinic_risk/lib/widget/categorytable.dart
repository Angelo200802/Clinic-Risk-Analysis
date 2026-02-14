import 'package:flutter/material.dart';

class CategoryTable extends StatelessWidget {
  final Map<String, Map<String, double>> categoryMetrics;
  final String selectedCategory;
  final Widget Function(Widget) buildGlassPanel;
  final Function(String) onTapCategory;
  final Widget Function(IconData, String) buildPanelHeader;

  const CategoryTable({
    super.key,
    required this.categoryMetrics,
    required this.selectedCategory,
    required this.buildGlassPanel,
    required this.onTapCategory,
    required this.buildPanelHeader,
  });

  Widget _buildCategorySelector(List<String> categories) {
    return Wrap(
      spacing: 8,
      children: categories.map((cat) {
        bool isSelected = selectedCategory == cat;
        return ChoiceChip(
          label: Text(cat.replaceAll("_", " ")),
          selected: isSelected,
          onSelected: (selected) {
            if (selected) {
              onTapCategory(cat);
            }
          },
          selectedColor: Colors.blueAccent,
          backgroundColor: Colors.white10,
          labelStyle: TextStyle(
            color: isSelected ? Colors.white : Colors.white60,
          ),
        );
      }).toList(),
    );
  }

  @override
  Widget build(BuildContext context) {
    List<String> categories = categoryMetrics.keys.toList();

    Map<String, dynamic> currentData = categoryMetrics[categories[0]]!;

    return buildGlassPanel(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              buildPanelHeader(Icons.analytics, "Performance by Category"),
              // Pulsanti di selezione (ToggleButtons o Dropdown)
              _buildCategorySelector(categories),
            ],
          ),
          const SizedBox(height: 20),
          SingleChildScrollView(
            scrollDirection:
                Axis.horizontal, // Rende la tabella scrollabile su mobile
            child: DataTable(
              headingRowColor: WidgetStateProperty.all(
                Colors.blueAccent.withOpacity(0.1),
              ),
              columnSpacing: 25,
              columns: const [
                DataColumn(
                  label: Text(
                    'GROUP',
                    style: TextStyle(color: Colors.blueAccent),
                  ),
                ),
                DataColumn(
                  label: Text(
                    'ACCURACY',
                    style: TextStyle(color: Colors.white70),
                  ),
                ),
                DataColumn(
                  label: Text(
                    'PRECISION',
                    style: TextStyle(color: Colors.white70),
                  ),
                ),
                DataColumn(
                  label: Text(
                    'RECALL',
                    style: TextStyle(color: Colors.white70),
                  ),
                ),
                DataColumn(
                  label: Text(
                    'ROC AUC',
                    style: TextStyle(color: Colors.white70),
                  ),
                ),
              ],
              rows: currentData.entries.map((entry) {
                final stats = entry.value;
                return DataRow(
                  cells: [
                    DataCell(
                      Text(
                        entry.key,
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ),
                    DataCell(
                      Text("${(stats['accuracy'] * 100).toStringAsFixed(1)}%"),
                    ),
                    DataCell(
                      Text("${(stats['precision'] * 100).toStringAsFixed(1)}%"),
                    ),
                    DataCell(
                      Text("${(stats['recall'] * 100).toStringAsFixed(1)}%"),
                    ),
                    DataCell(Text(stats['auc_roc'].toStringAsFixed(3))),
                  ],
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }
}
