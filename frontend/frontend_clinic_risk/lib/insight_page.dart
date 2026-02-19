import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'evaluation.dart';
import 'widget/radarchart.dart';
import 'livestream_page.dart';
import 'widget/scatter.dart';

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

            return LayoutBuilder(
              builder: (context, constraints) {
                bool isDesktop = constraints.maxWidth >= 1000;
                return SingleChildScrollView(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    children: [
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
                              child: MetabolicEffortChart(scatterData: points),
                            ),
                          ],
                        ),
                      ),
                      const Divider(color: Colors.white30),

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
              },
            );
          }
        },
      ),
    );
  }
}
