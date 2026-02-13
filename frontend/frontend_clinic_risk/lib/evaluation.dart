import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:frontend_clinic_risk/widget/confusionmatrix.dart';
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
  late Future<dynamic> confusionMatrixData;
  late Future<dynamic> metricsData;
  @override
  void initState() {
    super.initState();
    String apiUrl = Uri.parse(dotenv.env["BACKEND_BASE_API"]!).toString();
    debugPrint('API URL: $apiUrl');
    confusionMatrixData = fetchGet(apiUrl, 'evaluation/confusion_matrix');
    confusionMatrixData.then((data) {
      debugPrint('Fetched data: $data');
    });
    metricsData = fetchGet(apiUrl, 'evaluation/metrics');
    metricsData.then((data) {
      debugPrint('Fetched metrics data: $data');
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A1A), // Mantengo il tuo stile dark
      body: FutureBuilder<dynamic>(
        future: confusionMatrixData,
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
                snapshot.data['confusion_matrix'] as Map<String, dynamic>;

            return Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
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
                  ConfusionMatrixWidget(data: matrixMap),
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
