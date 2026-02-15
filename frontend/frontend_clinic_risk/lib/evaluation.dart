import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:frontend_clinic_risk/widget/confusionmatrix.dart';
import 'package:frontend_clinic_risk/widget/metrics.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:fl_chart/fl_chart.dart';
import 'livestream_page.dart';
import 'widget/roccurve.dart';

Function(String, String) fetchGet = (String url, String path) async {
  final endpoint = Uri.parse('$url/$path');

  try {
    final response = await http.get(endpoint);
    debugPrint('Response status: ${response.statusCode}');
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load data');
    }
  } catch (e) {
    throw Exception('Error fetching data: $e');
  }
};

Widget buildPanelHeader(IconData icon, String title) {
  return Row(
    children: [
      Icon(icon, color: Colors.blueAccent, size: 20),
      const SizedBox(width: 10),
      Text(
        title,
        style: const TextStyle(
          color: Colors.white70,
          fontSize: 16,
          fontWeight: FontWeight.w600,
        ),
      ),
    ],
  );
}

Widget _buildStratifiedTable(
  Map<String, dynamic> allData,
  String selectedCategory,
  Function(String) onTapCategory,
) {
  // Otteniamo le categorie disponibili (Gender, Age_Range, etc.)
  List<String> categories = allData.keys.toList();

  // Otteniamo i dati per la categoria selezionata (es. Male/Female)
  Map<String, dynamic> currentData = allData[selectedCategory];

  return buildGlassPanel(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            buildPanelHeader(Icons.analytics, "Performance per Categoria"),
            const SizedBox(height: 20),
            _buildCategorySelector(categories, selectedCategory, onTapCategory),
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
                  'SOTTOGRUPPO',
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
                label: Text('RECALL', style: TextStyle(color: Colors.white70)),
              ),
              DataColumn(
                label: Text('ROC AUC', style: TextStyle(color: Colors.white70)),
              ),
            ],
            rows: currentData.entries.map((entry) {
              final stats = entry.value;
              return DataRow(
                cells: [
                  DataCell(
                    Text(
                      entry.key,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  DataCell(
                    Text(
                      "${(stats['accuracy'] * 100).toStringAsFixed(1)}%",
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  DataCell(
                    Text(
                      "${(stats['precision'] * 100).toStringAsFixed(1)}%",
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  DataCell(
                    Text(
                      "${(stats['recall'] * 100).toStringAsFixed(1)}%",
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  DataCell(
                    Text(
                      stats['auc_roc'].toStringAsFixed(3),
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
              );
            }).toList(),
          ),
        ),
      ],
    ),
  );
}

// Widget per i bottoni di selezione
Widget _buildCategorySelector(
  List<String> categories,
  String selectedCategory,
  Function(String) onTapCategory,
) {
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
        backgroundColor: Colors.red,
        labelStyle: TextStyle(
          color: isSelected ? Colors.white : Colors.white60,
        ),
      );
    }).toList(),
  );
}

class EvaluationPage extends StatefulWidget {
  const EvaluationPage({super.key});

  @override
  State<EvaluationPage> createState() => _EvaluationPageState();
}

class _EvaluationPageState extends State<EvaluationPage> {
  late Future<List<dynamic>> combinedData;
  String selectedCategory = "Gender";

  @override
  void initState() {
    super.initState();
    String apiUrl = Uri.parse(dotenv.env["BACKEND_BASE_API"]!).toString();
    combinedData = Future.wait([
      fetchGet(apiUrl, 'evaluation/confusion_matrix'),
      fetchGet(apiUrl, 'evaluation/metrics'),
      fetchGet(apiUrl, "evaluation/evaluate_by_category"),
    ]);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A1A), // Mantengo il tuo stile dark
      body: FutureBuilder<List<dynamic>>(
        future: combinedData,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (snapshot.hasError) {
            return Center(child: Text("Errore: ${snapshot.error}"));
          }

          if (snapshot.hasData) {
            // Accediamo alla mappa interna come restituita dal tuo backend
            final matrixMap =
                snapshot.data![0]['confusion_matrix'] as Map<String, dynamic>;
            final rawMetrics = snapshot.data![1];
            final categoryMetrics = snapshot.data![2] as Map<String, dynamic>;
            debugPrint(
              'Raw metrics: $rawMetrics',
            ); // Debug per vedere i dati ricevuti
            final metrics = Metrics.fromJson(rawMetrics);
            List<FlSpot> points = metrics.rocCurve
                .map((point) => FlSpot(point['fpr']!, point['tpr']!))
                .toList();
            return LayoutBuilder(
              builder: (context, constraints) {
                double screenWidth = constraints.maxWidth;
                bool isDesktop = screenWidth >= 1000;
                return SingleChildScrollView(
                  physics: const AlwaysScrollableScrollPhysics(),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24.0,
                    vertical: 40.0,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.analytics_outlined,
                        color: Colors.blueAccent,
                        size: 40,
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        "MODEL EVALUATION",
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 30),
                      if (isDesktop)
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // COLONNA SINISTRA: Metriche + Matrice
                            Expanded(
                              flex:
                                  1, // Puoi regolare il flex per cambiare le proporzioni
                              child: Column(
                                children: [
                                  // PANNELLO 1: METRICHE
                                  buildGlassPanel(
                                    child: Column(
                                      children: [
                                        buildPanelHeader(
                                          Icons.speed,
                                          "Performance",
                                        ),
                                        const SizedBox(height: 20),
                                        MetricsDashboard(metrics: metrics),
                                      ],
                                    ),
                                  ),
                                  const SizedBox(height: 24),
                                  // PANNELLO 2: MATRICE DI CONFUSIONE
                                  buildGlassPanel(
                                    child: Column(
                                      children: [
                                        buildPanelHeader(
                                          Icons.grid_on,
                                          "Matrice di Confusione",
                                        ),
                                        const SizedBox(height: 20),
                                        SizedBox(
                                          width: double.infinity,
                                          child: ConfusionMatrixWidget(
                                            data: matrixMap,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),

                            const SizedBox(width: 24),

                            _buildStratifiedTable(
                              categoryMetrics,
                              selectedCategory,
                              (category) {
                                setState(() {
                                  selectedCategory = category;
                                });
                              },
                            ),

                            const SizedBox(width: 24),
                            // COLONNA DESTRA: GRAFICO ROC
                            Expanded(
                              flex: 1,
                              child: buildGlassPanel(
                                child: Column(
                                  children: [
                                    buildPanelHeader(
                                      Icons.show_chart,
                                      "Curva ROC",
                                    ),
                                    const SizedBox(height: 20),
                                    // Il grafico si adatterà all'altezza disponibile
                                    RocCurveChart(points: points),
                                    const SizedBox(height: 20),
                                    const Text(
                                      "L'area sotto la curva (AUC) indica la capacità discriminante del modello. Un valore vicino a 1.0 è ottimale.",
                                      style: TextStyle(
                                        color: Colors.white54,
                                        fontSize: 12,
                                        fontStyle: FontStyle.italic,
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        )
                      else ...[
                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(Icons.speed, "Performance"),
                              const SizedBox(height: 20),
                              MetricsDashboard(metrics: metrics),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),
                        // Layout Mobile: Pannelli uno sotto l'altro
                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(
                                Icons.grid_on,
                                "Matrice di Confusione",
                              ),
                              const SizedBox(height: 20),
                              ConfusionMatrixWidget(data: matrixMap),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),
                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(Icons.show_chart, "Curva ROC"),
                              const SizedBox(height: 20),
                              RocCurveChart(points: points),
                            ],
                          ),
                        ),

                        const SizedBox(height: 24),
                        _buildStratifiedTable(
                          categoryMetrics,
                          selectedCategory,
                          (category) {
                            setState(() {
                              selectedCategory = category;
                            });
                          },
                        ),
                      ],

                      const SizedBox(height: 40),
                    ],
                  ),
                );
              },
            );
          }

          return const Center(child: Text("Nessun dato disponibile"));
        },
      ),
    );
  }
}
