import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:frontend_clinic_risk/widget/confusionmatrix.dart';
import 'package:frontend_clinic_risk/widget/metrics.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';

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

class EvaluationPage extends StatefulWidget {
  const EvaluationPage({super.key});

  @override
  State<EvaluationPage> createState() => _EvaluationPageState();
}

class _EvaluationPageState extends State<EvaluationPage> {
  late Future<List<dynamic>> combinedData;

  @override
  void initState() {
    super.initState();
    String apiUrl = Uri.parse(dotenv.env["BACKEND_BASE_API"]!).toString();
    combinedData = Future.wait([
      fetchGet(apiUrl, 'evaluation/confusion_matrix'),
      fetchGet(apiUrl, 'evaluation/metrics'),
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
            final metrics = Metrics.fromJson(rawMetrics);

            return SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
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

                  // Avvolgiamo le metriche in un Flexible o lasciamo che shrinkWrap faccia il suo dovere
                  MetricsDashboard(metrics: metrics),
                  const SizedBox(height: 40),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 20),

                  const Text(
                    "CONFUSION MATRIX",
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 20),
                  SizedBox(
                    width:
                        MediaQuery.of(context).size.width -
                        48, // Sottraiamo il padding orizzontale (24+24)
                    child: ConfusionMatrixWidget(data: matrixMap),
                  ),
                  /*
                  // Fix per matrici troppo larghe
                  LayoutBuilder(
                    builder: (context, constraints) {
                      return SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: ConstrainedBox(
                          constraints: BoxConstraints(
                            minWidth: constraints.maxWidth,
                          ),
                          child: ConfusionMatrixWidget(data: matrixMap),
                        ),
                      );
                    },
                  ),*/
                ],
              ),
            );
          }
          return const SizedBox();
        },
      ),
    );
  }
}
