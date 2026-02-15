import 'package:flutter/material.dart';
import 'widget/riskbarchart.dart';
import 'evaluation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'livestream_page.dart';
import 'evaluation.dart';
import 'widget/correlationmatrix.dart';

class AnalyticsPage extends StatefulWidget {
  const AnalyticsPage({super.key});

  @override
  State<AnalyticsPage> createState() => _AnalyticsPageState();
}

class _AnalyticsPageState extends State<AnalyticsPage> {
  late Future<List<dynamic>> combinedData;

  @override
  void initState() {
    super.initState();
    String apiUrl = Uri.parse(dotenv.env["BACKEND_BASE_API"]!).toString();
    combinedData = Future.wait([
      fetchGet(apiUrl, 'stats/summary'),
      fetchGet(apiUrl, 'stats/age_risk'),
      fetchGet(apiUrl, 'stats/gender_risk'),
      fetchGet(apiUrl, 'stats/bmi_risk'),
      fetchGet(apiUrl, 'stats/correlation_matrix'),
    ]);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A1A).withOpacity(0.4),
      body: FutureBuilder<List<dynamic>>(
        future: combinedData,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return const Center(child: Text('No data available'));
          } else {
            final summary = snapshot.data![0];
            final riskByAge = snapshot.data![1];
            final riskByGender = snapshot.data![2];
            final riskByBmi = snapshot.data![3];
            final correlationMatrix =
                (snapshot.data![4]['correlation_matrix'] as List).map((row) {
                  return (row as List)
                      .map((value) => (value as num).toDouble())
                      .toList();
                }).toList();
            final labels = snapshot.data![4]['columns'];
            debugPrint("Correlation Matrix: $correlationMatrix");
            debugPrint("Labels: $labels");
            return LayoutBuilder(
              builder: (context, constraints) {
                bool isDesktop = constraints.maxWidth >= 1000;

                return SingleChildScrollView(
                  padding: const EdgeInsets.all(
                    24,
                  ), // Un po' di respiro ai bordi
                  child: Column(
                    children: [
                      if (isDesktop)
                        // --- LAYOUT DESKTOP: Age e Gender affiancati ---
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(
                              child: buildGlassPanel(
                                child: Column(
                                  children: [
                                    buildPanelHeader(
                                      Icons.fitness_center,
                                      "Risk by BMI",
                                    ),
                                    const SizedBox(height: 20),
                                    SizedBox(
                                      height: 300,
                                      width: double.infinity,
                                      child: DynamicRiskBarChart(
                                        rawData: riskByBmi['data'],
                                        categoryKey: 'BMI_Category',
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                            const SizedBox(
                              width: 24,
                            ), // Spazio tra i due pannelli
                            Expanded(
                              child: buildGlassPanel(
                                child: Column(
                                  children: [
                                    buildPanelHeader(
                                      Icons.male,
                                      "Risk by Gender",
                                    ),
                                    const SizedBox(height: 20),
                                    SizedBox(
                                      height: 300,
                                      child: DynamicRiskBarChart(
                                        rawData: riskByGender['data'],
                                        categoryKey: 'Gender',
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        )
                      else
                        Column(
                          children: [
                            buildGlassPanel(
                              child: Column(
                                children: [
                                  buildPanelHeader(
                                    Icons.fitness_center,
                                    "Risk by BMI",
                                  ),
                                  const SizedBox(height: 20),
                                  SizedBox(
                                    height: 300,
                                    width: double.infinity,
                                    child: DynamicRiskBarChart(
                                      rawData: riskByBmi['data'],
                                      categoryKey: 'BMI_Category',
                                    ),
                                  ),
                                ],
                              ),
                            ),

                            const SizedBox(height: 20),
                            buildGlassPanel(
                              child: Column(
                                children: [
                                  buildPanelHeader(
                                    Icons.male,
                                    "Risk by Gender",
                                  ),
                                  const SizedBox(height: 20),
                                  SizedBox(
                                    height: 300,
                                    child: DynamicRiskBarChart(
                                      rawData: riskByGender['data'],
                                      categoryKey: 'Gender',
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),

                      const SizedBox(height: 24),

                      buildGlassPanel(
                        child: Column(
                          children: [
                            buildPanelHeader(Icons.bar_chart, "Risk by Age"),
                            const SizedBox(height: 20),
                            SizedBox(
                              height:
                                  300, // Altezza fissa, larghezza dinamica (Expanded)
                              child: DynamicRiskBarChart(
                                rawData: riskByAge['data'],
                                categoryKey: 'Decade',
                              ),
                            ),
                          ],
                        ),
                      ),

                      buildGlassPanel(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            buildPanelHeader(
                              Icons.apps,
                              "Feature Correlation Heatmap",
                            ),
                            const SizedBox(height: 20),
                            SingleChildScrollView(
                              scrollDirection: Axis
                                  .horizontal, // Permette lo scroll se le variabili sono tante
                              child: CorrelationMatrixWidget(
                                labels: labels.cast<String>(),
                                matrix: correlationMatrix.cast<List<double>>(),
                              ),
                            ),
                          ],
                        ),
                      ),

                      // --- BMI: Sempre a tutta larghezza (o puoi affiancarlo ad altro) ---
                    ],
                  ),
                );
              },
            );
          }
        },
      ),
    );
  }
}
