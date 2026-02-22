import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'evaluation.dart';
import 'livestream_page.dart';
import 'widget/radarchart.dart';
import 'widget/scatter.dart';
import 'widget/classification.dart';

// Helper per la legenda
Widget _legendItem(String label, Color color) {
  return Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Container(
        width: 12,
        height: 12,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      ),
      const SizedBox(width: 8),
      Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
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
      fetchGet(url, 'clinic/k_nearest?fraction=0.01&radius=3.0'),
    ]);
  }

  @override
  Widget build(BuildContext context) {
    // Definiamo un'altezza standard per le card in base allo schermo
    final double panelHeight = MediaQuery.of(context).size.height * 0.6;
    final double mobilePanelHeight =
        panelHeight * 0.8; // Altezza ridotta per mobile

    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      body: FutureBuilder(
        future: combinedData,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(
              child: CircularProgressIndicator(color: Colors.blueAccent),
            );
          } else if (snapshot.hasError) {
            return Center(
              child: Text(
                'Error: ${snapshot.error}',
                style: const TextStyle(color: Colors.red),
              ),
            );
          } else {
            final cardiacStressRank = snapshot.data![2]['data'];
            final derivedIndices = snapshot.data![1]['data'];

            return LayoutBuilder(
              builder: (context, constraints) {
                bool isDesktop = constraints.maxWidth >= 1100;

                return SingleChildScrollView(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    children: [
                      // Header della pagina
                      const Icon(
                        Icons.insights_outlined,
                        color: Colors.blueAccent,
                        size: 48,
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        "Clinical Intelligence",
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 32),

                      if (isDesktop) ...[
                        // LAYOUT DESKTOP: 3 COLONNE ALLINEATE
                        SizedBox(
                          height: panelHeight,
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Expanded(
                                child: _buildPanelWrapper(
                                  Icons.favorite_border,
                                  "Cardiac Stress Rank",
                                  CardiacStressRanking(data: cardiacStressRank),
                                ),
                              ),
                              const SizedBox(width: 20),
                              Expanded(
                                child: _buildPanelWrapper(
                                  Icons.warning_amber_rounded,
                                  "Obesity Mismatch",
                                  ObesityMismatchList(
                                    data: snapshot.data![3]['data'],
                                  ),
                                ),
                              ),
                              const SizedBox(width: 20),
                              Expanded(
                                child: _buildPanelWrapper(
                                  Icons.visibility_off_outlined,
                                  "Occult Shock Alert",
                                  OccultShockAlert(
                                    data: snapshot.data![4]['data'],
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),
                        // SECONDA RIGA: RADAR E SCATTER
                        SizedBox(
                          height: panelHeight,
                          child: Row(
                            children: [
                              Expanded(
                                child: buildGlassPanel(
                                  child: Column(
                                    children: [
                                      buildPanelHeader(
                                        Icons.radar,
                                        "Emodynamic Fingerprint",
                                      ),
                                      const Spacer(),
                                      Expanded(
                                        child: ClinicalRadarChart(
                                          radarData: derivedIndices,
                                        ),
                                      ),
                                      const SizedBox(height: 16),
                                      Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.center,
                                        children: [
                                          _legendItem(
                                            "High Risk",
                                            Colors.redAccent,
                                          ),
                                          const SizedBox(width: 20),
                                          _legendItem(
                                            "Low Risk",
                                            Colors.greenAccent,
                                          ),
                                        ],
                                      ),
                                      const Spacer(),
                                    ],
                                  ),
                                ),
                              ),
                              const SizedBox(width: 20),
                              Expanded(
                                child: buildGlassPanel(
                                  child: Column(
                                    children: [
                                      buildPanelHeader(
                                        Icons.bubble_chart_outlined,
                                        "K-Nearest Risk Proximity",
                                      ),
                                      const SizedBox(height: 16),
                                      Expanded(
                                        child: ClinicScatterChart(
                                          scatterData:
                                              snapshot.data![5]['data'],
                                          xKey: "Derived_MAP",
                                          yKey: "Derived_BMI",
                                          xLabel: "Derived MAP (mmHg)",
                                          yLabel: "Derived BMI",
                                          highRiskColor: Colors.redAccent,
                                          lowRiskColor: Colors.greenAccent,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              const SizedBox(height: 24),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),
                        SizedBox(
                          height: panelHeight,
                          child: buildGlassPanel(
                            child: Column(
                              children: [
                                buildPanelHeader(
                                  Icons.scatter_plot,

                                  "Metabolic Shock Index Scatter Plot",
                                ),

                                const SizedBox(height: 16),
                                Expanded(
                                  child: ClinicScatterChart(
                                    scatterData: snapshot.data![0]['data'],

                                    xKey: "ShockIndex",

                                    yKey: "PulsePressureIndex",

                                    xLabel: "Shock Index (HR / SBP)",

                                    yLabel: "Pulse Pressure Index (PP / HR)",

                                    highRiskColor: Colors.orangeAccent,

                                    lowRiskColor: Colors.blue,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ] else ...[
                        // LAYOUT MOBILE: COLONNA SINGOLA
                        _buildMobilePanel(
                          Icons.class_,
                          "Cardiac Stress",
                          CardiacStressRanking(data: cardiacStressRank),
                          mobilePanelHeight,
                        ),
                        const SizedBox(height: 20),
                        _buildMobilePanel(
                          Icons.warning,
                          "Obesity Mismatch",
                          ObesityMismatchList(data: snapshot.data![3]['data']),
                          mobilePanelHeight,
                        ),
                        const SizedBox(height: 20),
                        _buildMobilePanel(
                          Icons.visibility_off,
                          "Occult Shock",
                          OccultShockAlert(data: snapshot.data![4]['data']),
                          mobilePanelHeight,
                        ),
                        const SizedBox(height: 20),
                        // Radar e Scatter in mobile hanno bisogno di AspectRatio
                        buildGlassPanel(
                          child: Column(
                            children: [
                              buildPanelHeader(Icons.radar, "Emodynamic Radar"),
                              AspectRatio(
                                aspectRatio: 1,
                                child: ClinicalRadarChart(
                                  radarData: derivedIndices,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
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

  // Wrapper per mantenere i pannelli puliti
  Widget _buildPanelWrapper(IconData icon, String title, Widget child) {
    return buildGlassPanel(
      child: Column(
        children: [
          buildPanelHeader(icon, title),
          const SizedBox(height: 16),
          Expanded(child: child),
        ],
      ),
    );
  }

  // Wrapper per Mobile
  Widget _buildMobilePanel(
    IconData icon,
    String title,
    Widget child,
    double height,
  ) {
    return SizedBox(
      height: height,
      child: _buildPanelWrapper(icon, title, child),
    );
  }
}
