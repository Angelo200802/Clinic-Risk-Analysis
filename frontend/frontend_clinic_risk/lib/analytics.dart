import 'package:flutter/material.dart';
import 'widget/riskbarchart.dart';
import 'evaluation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'livestream_page.dart';
import 'evaluation.dart';
import 'widget/statvisualizer.dart';
import 'widget/correlationmatrix.dart';
import 'widget/stressmap.dart';

class AnalyticsPage extends StatefulWidget {
  const AnalyticsPage({super.key});

  @override
  State<AnalyticsPage> createState() => _AnalyticsPageState();
}

Map<String, Map<String, dynamic>> signsMap = {
  "heart_rate": {"unit": "BPM", "label": "HR", "payload": null},
  "respiratory_rate": {
    "unit": "Breaths/min",
    "label": "Respiratory Rate",
    "payload": null,
  },
  "oxygen_saturation": {"unit": "%", "label": "SpO2", "payload": null},
  "body_temperature": {
    "unit": "°C",
    "label": "Body Temperature",
    "payload": null,
  },
  "derived_pulse_pressure": {"unit": "mmHg", "label": "PP", "payload": null},
  "derived_hrv": {"unit": "ms", "label": "HRV", "payload": null},
  "derived_map": {"unit": "mmHg", "label": "MAP", "payload": null},
};

Widget buildStats(String label, String unit, dynamic data) {
  return StatRangeVisualizer(
    title: label,
    min: data['min'],
    max: data['max'],
    mean: data['mean'],
    stdDev: data['stddev'],
    unit: unit,
    totalSamples: data['count'],
  );
}

class _AnalyticsPageState extends State<AnalyticsPage> {
  late Future<List<dynamic>> combinedData;
  String state = "heart_rate";

  @override
  void initState() {
    super.initState();
    String apiUrl = Uri.parse(dotenv.env["BACKEND_BASE_API"]!).toString();
    combinedData = Future.wait([
      fetchGet(apiUrl, 'stats/age_risk'),
      fetchGet(apiUrl, 'stats/gender_risk'),
      fetchGet(apiUrl, 'stats/bmi_risk'),
      fetchGet(apiUrl, 'stats/correlation_matrix'),
      fetchGet(apiUrl, 'stats?signs=heart_rate'),
      fetchGet(apiUrl, 'stats?signs=respiratory_rate'),
      fetchGet(apiUrl, 'stats?signs=oxygen_saturation'),
      fetchGet(apiUrl, 'stats?signs=body_temperature'),
      fetchGet(apiUrl, 'stats?signs=derived_pulse_pressure'),
      fetchGet(apiUrl, 'stats?signs=derived_hrv'),
      fetchGet(apiUrl, 'stats?signs=derived_map'),
      fetchGet(apiUrl, 'stats/demographic_stress_map'),
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
            final riskByAge = snapshot.data![0];
            final riskByGender = snapshot.data![1];
            final riskByBmi = snapshot.data![2];
            final correlationMatrix =
                (snapshot.data![3]['correlation_matrix'] as List).map((row) {
                  return (row as List)
                      .map((value) => (value as num).toDouble())
                      .toList();
                }).toList();
            final labels = snapshot.data![3]['columns'];
            signsMap['heart_rate']!['payload'] = snapshot.data![4]!;
            signsMap['respiratory_rate']!['payload'] = snapshot.data![5];
            signsMap['oxygen_saturation']!['payload'] = snapshot.data![6];
            signsMap['body_temperature']!['payload'] = snapshot.data![7];
            signsMap['derived_pulse_pressure']!['payload'] = snapshot.data![8];
            signsMap['derived_hrv']!['payload'] = snapshot.data![9];
            signsMap['derived_map']!['payload'] = snapshot.data![10];

            final stressMap = snapshot.data![11]['data']
                .cast<Map<String, dynamic>>();

            return LayoutBuilder(
              builder: (context, constraints) {
                bool isDesktop = constraints.maxWidth >= 1000;

                return SingleChildScrollView(
                  padding: const EdgeInsets.all(
                    24,
                  ), // Un po' di respiro ai bordi
                  child: Column(
                    children: [
                      const Icon(
                        Icons.history_edu_outlined,
                        color: Colors.blueAccent,
                        size: 40,
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        "Analytics",
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 30),
                      buildGlassPanel(
                        child: Column(
                          children: [
                            Row(
                              children: [
                                buildPanelHeader(
                                  Icons.fitness_center,
                                  "Statistics",
                                ),
                                const SizedBox(width: 16),
                                buildCategorySelector(
                                  signsMap.keys
                                      .toList()
                                      .map(
                                        (e) => signsMap[e]!['label']! as String,
                                      )
                                      .toList(),
                                  signsMap[state]!['label']! as String,
                                  (newState) {
                                    setState(
                                      () => state = signsMap.keys.firstWhere(
                                        (k) =>
                                            signsMap[k]!['label'] == newState,
                                      ),
                                    );
                                  },
                                ),
                              ],
                            ),
                            const SizedBox(height: 20),
                            buildStats(
                              state.replaceAll(RegExp(r'_'), " "),
                              signsMap[state]!['unit']!,
                              signsMap[state]!['payload'],
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 24),
                      if (isDesktop)
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
                            const SizedBox(width: 24),
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
                      const SizedBox(height: 24),

                      buildGlassPanel(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            buildPanelHeader(
                              Icons.map,
                              "Demographic Stress Map",
                            ),
                            const SizedBox(height: 20),
                            DemographicStressMap(data: stressMap),
                          ],
                        ),
                      ),

                      const SizedBox(height: 24),
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
