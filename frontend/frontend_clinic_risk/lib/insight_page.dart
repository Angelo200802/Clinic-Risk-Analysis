import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'evaluation.dart';
import 'widget/radarchart.dart';
import 'livestream_page.dart';
import 'widget/scatter.dart';
import 'widget/classification.dart';

Widget _legendItem(String label, Color color) {
  return Row(
    children: [
      Container(width: 12, height: 12, color: color),
      const SizedBox(width: 5),
      Text(label, style: const TextStyle(color: Colors.white)),
    ],
  );
}

class InsigthPage extends StatefulWidget {
  const InsigthPage({super.key});

  @override
  State<InsigthPage> createState() => _InsigthPageState();
}

class _InsigthPageState extends State<InsigthPage> {
  final String url = Uri.parse(dotenv.env["BACKEND_BASE_API"]!).toString();
  late Future<List<dynamic>> combinedData;

  @override
  void initState() {
    super.initState();
    combinedData = Future.wait([
      fetchGet(url, 'clinic/metabolic_shockindex?fraction=0.01'),
      fetchGet(url, 'clinic/derived_indices'),
      fetchGet(url, 'clinic/top_cardiac_stress'),
      fetchGet(url, 'clinic/obesity_mismatch'),
      fetchGet(url, 'clinic/occult_shock'),
    ]);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A1A).withOpacity(0.4),
      body: FutureBuilder(
        future: combinedData,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else {
            List<dynamic> points = snapshot.data![0]['data'];
            List<dynamic> derivedIndices = snapshot.data![1]['data'];
            List<dynamic> cardiacStressRank = snapshot.data![2]['data'];

            return LayoutBuilder(
              builder: (context, constraints) {
                bool isDesktop = constraints.maxWidth >= 1000;

                if (isDesktop) {
                  return SingleChildScrollView(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      children: [
                        const Icon(
                          Icons.insights_outlined,
                          color: Colors.blueAccent,
                          size: 40,
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          "Insights",
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 30),
                        Row(
                          children: [
                            Expanded(
                              child: buildGlassPanel(
                                child: Column(
                                  children: [
                                    buildPanelHeader(
                                      Icons.class_,
                                      "Top 5 Cardiac Stress Patients",
                                    ),
                                    const SizedBox(height: 16),
                                    CardiacStressRanking(
                                      data: cardiacStressRank,
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
                                      Icons.warning_amber_outlined,
                                      "Obesity Mismatch Cases",
                                    ),
                                    const SizedBox(height: 16),
                                    ObesityMismatchList(
                                      data: snapshot.data![3]['data'],
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
                                      Icons.visibility_off,
                                      "Occult Shock Cases",
                                    ),
                                    const SizedBox(height: 16),
                                    OccultShockAlert(
                                      data: snapshot.data![4]['data'],
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),

                        const SizedBox(height: 24),
                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(
                                Icons.scatter_plot,
                                "Metabolic Shock Index Scatter Plot",
                              ),
                              const SizedBox(height: 16),
                              SizedBox(
                                height: 300,
                                child: MetabolicEffortChart(
                                  scatterData: points,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),

                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(
                                Icons.radar_sharp,
                                "Derived Indices Radar Chart",
                              ),
                              const SizedBox(height: 16),
                              ClinicalRadarChart(radarData: derivedIndices),
                              const SizedBox(height: 16),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  _legendItem("High Risk", Colors.redAccent),
                                  const SizedBox(width: 20),
                                  _legendItem("Low Risk", Colors.greenAccent),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  );
                } else {
                  return SingleChildScrollView(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      children: [
                        const Icon(
                          Icons.insights_outlined,
                          color: Colors.blueAccent,
                          size: 40,
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          "Insights",
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
                              buildPanelHeader(
                                Icons.class_,
                                "Top 5 Cardiac Stress Patients",
                              ),
                              const SizedBox(height: 16),
                              CardiacStressRanking(data: cardiacStressRank),
                            ],
                          ),
                        ),

                        const SizedBox(height: 24),

                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(
                                Icons.warning_amber_outlined,
                                "Obesity Mismatch Cases",
                              ),
                              const SizedBox(height: 16),
                              ObesityMismatchList(
                                data: snapshot.data![3]['data'],
                              ),
                            ],
                          ),
                        ),

                        const SizedBox(height: 24),

                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(
                                Icons.visibility_off,
                                "Occult Shock Cases",
                              ),
                              const SizedBox(height: 16),
                              OccultShockAlert(data: snapshot.data![4]['data']),
                            ],
                          ),
                        ),

                        const SizedBox(height: 24),
                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(
                                Icons.scatter_plot,
                                "Metabolic Shock Index Scatter Plot",
                              ),
                              const SizedBox(height: 16),
                              SizedBox(
                                height: 300,
                                child: MetabolicEffortChart(
                                  scatterData: points,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),

                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(
                                Icons.radar_sharp,
                                "Derived Indices Radar Chart",
                              ),
                              const SizedBox(height: 16),
                              ClinicalRadarChart(radarData: derivedIndices),
                              const SizedBox(height: 16),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  _legendItem("High Risk", Colors.redAccent),
                                  const SizedBox(width: 20),
                                  _legendItem("Low Risk", Colors.greenAccent),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  );
                }
              },
            );
          }
        },
      ),
    );
  }
}
