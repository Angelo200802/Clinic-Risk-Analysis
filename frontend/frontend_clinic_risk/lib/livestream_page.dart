import 'dart:async';
import 'package:flutter/material.dart' hide Chip;
import 'package:frontend_clinic_risk/types/indexclass.dart';
import 'package:frontend_clinic_risk/widget/ai_widget.dart';
import 'package:frontend_clinic_risk/widget/badgeswidget.dart';
import 'package:frontend_clinic_risk/widget/feature.dart';
import 'package:frontend_clinic_risk/widget/radarchart.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'widget/patient_card.dart';
import 'types/trend.dart';
import 'types/sensorUpdate.dart';
import 'types/pattern.dart';
import 'widget/trend_graph.dart';
import 'widget/iconbuttonrow.dart';
import 'widget/circularsummary.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

const dynamic payload = {
  'sensor_update': {
    'Heart Rate': 94,
    'Respiratory Rate': 13,
    'Oxygen Saturation': 96.00851486564063,
    'Systolic Blood Pressure': 133.0,
    'Diastolic Blood Pressure': 87.0,
    'Body Temperature': 36.78481837007358,
    'Age': 48,
    'Gender': 'Male',
    'Weight (kg)': 59.554568255345046,
    'Height (m)': 1.5244914505173788,
    'Derived_MAP': 102.33333333333333,
    'Derived_HRV': 0.05515034978780148,
    'Derived_BMI': 25.625071995826932,
    'Derived_Pulse_Pressure': 46.0,
    'Prediction': 'High Risk',
    'Patient ID': 12655,
    'Timestamp': '2026-02-11T19:23:58.301511',
  },
  'trend_update': {
    'Patient ID': 12655,
    'risk_ratio': 1.0,
    'avg_hr': 94.5,
    'avg_sbp': 133.0,
    'avg_dbp': 87.0,
    'avg_rr': 13.0,
    'avg_spo2': 96.00851486564063,
    'avg_temp': 36.78481837007358,
    'avg_map': 102.33333333333333,
    'avg_pp': 46.0,
    'avg_hrv': 0.05515034978780148,
    'std_hr': 0.7071067811865476,
    'n_samples': 2,
    'bmi_class': 'OVERWEIGHT',
    'hr_pct': 0.0,
    'rr_pct': 0.0,
    'spo2_pct': 0.0,
    'pp_pct': 0.0,
    'map_pct': 0.0,
    'progressive_hemo_deterioration': 0,
    'start': '2026-02-11T19:23:30',
    'end': '2026-02-11T19:24:30',
    'Timestamp': '2026-02-11T19:23:58.301511',
  },
  'index': {
    'shock_index': 0.7105263157894737,
    'modified_shock_index': 0.9234527687296418,
    'age_index': 34.10526315789474,
    'diastolic_shock_index': 1.0862068965517242,
    'rate_pp': 12568.5,
    'pp_index': 0.48677248677248675,
  },
  'pattern': {
    'progressive_hemo_deterioration': 1,
    'progressive_resp_failure_pattern': 1,
    'dynamic_sepsis_pattern': 0,
  },
};

Function(Trend) getFeatureInsights = (Trend trend) => [
  FeatureInsight(
    name: "Heart Rate Δ%",
    deltaPercentage: trend.hrPct,
    value: trend.hrPct,
  ),
  FeatureInsight(
    name: "Respiratory Rate Δ%",
    deltaPercentage: trend.rrPct,
    value: trend.rrPct,
  ),
  FeatureInsight(
    name: "SpO₂ Δ%",
    deltaPercentage: trend.spo2Pct,
    value: trend.spo2Pct,
  ),
  FeatureInsight(
    name: "Pulse Pressure Δ%",
    deltaPercentage: trend.ppPct,
    value: trend.ppPct,
  ),
  FeatureInsight(
    name: "Mean Arterial Pressure Δ%",
    deltaPercentage: trend.mapPct,
    value: trend.mapPct,
  ),
];

Function(double) getColor = (value) {
  if (value < 36.0) return Colors.blueAccent;
  if (value <= 37.2) return Colors.greenAccent;
  if (value <= 38.2) return Colors.orangeAccent;
  return Colors.redAccent;
};

Widget buildGlassPanel({required Widget child}) {
  return Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: const Color(0xFF1A1A1A).withOpacity(0.4),
      borderRadius: BorderRadius.circular(20),
      border: Border.all(color: Colors.white10),
      boxShadow: [
        BoxShadow(
          color: Colors.black.withOpacity(0.2),
          blurRadius: 10,
          offset: const Offset(0, 4),
        ),
      ],
    ),
    child: child,
  );
}

Widget _buildRiskBadgesPanel(Map<String, dynamic> riskData) {
  debugPrint(riskData.toString());
  return buildGlassPanel(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              "RISK INDICATORS",
              style: TextStyle(
                color: Colors.white54,
                fontSize: 10,
                letterSpacing: 1.2,
              ),
            ),
            Icon(Icons.warning_amber_rounded, color: Colors.white24, size: 14),
          ],
        ),
        const Divider(color: Colors.white10, height: 20),
        const SizedBox(height: 16),
        Center(
          child: Column(
            children: [
              Badges(
                label: "HEMO DETERIORATION",
                icon: Icons.bolt,
                color: Colors.redAccent,
                isActive: riskData['hemo_deterioration'] ?? false,
              ),
              const SizedBox(height: 12),
              Badges(
                label: "RESP. FAILURE",
                icon: Icons.coronavirus,
                color: Colors.orangeAccent,
                isActive: riskData['resp_failure'] ?? false,
              ),
              const SizedBox(height: 12),
              Badges(
                label: "DYNAMIC SEPSIS",
                icon: Icons.warning_amber,
                color: Colors.yellowAccent,
                isActive: riskData['dynamic_sepsis'] ?? false,
              ),
            ],
          ),
        ),
      ],
    ),
  );
}

class LivestreamPage extends StatefulWidget {
  const LivestreamPage({super.key});

  @override
  State<LivestreamPage> createState() => _LivestreamPageState();
}

class Record {
  SensorUpdate sensorUpdate;
  CalculatedIndex index;
  Pattern pattern;

  Record({
    required this.sensorUpdate,
    required this.index,
    required this.pattern,
  });
}

class _LivestreamPageState extends State<LivestreamPage> {
  late WebSocketChannel _channel;
  StreamSubscription? _subscription;
  Timer? _reconnectTimer;
  bool _isConnected = false;
  bool _isConnecting = false;
  Map<int, Record> allPatients = {};
  Map<int, List<Trend>> allTrends = {};
  SensorUpdate? _lastUpdate;
  int? selectedPatientId;
  String label = "Heart Rate";
  bool _showAiPanel = false;
  final Map<int, StringBuffer> _aiStreams = {}; // buffer per paziente
  final Map<int, String> _aiResponses = {}; // risposta completa finale
  bool _aiStreaming = false;
  Timer? _aiTypingTimer;
  int _aiTypingIndex = 0;
  String _aiFullText = '';

  Future<Map<String, dynamic>> _requestAiExplanation(int patientId) async {
    final uri = Uri.parse('http://localhost:8081/explain/$patientId');
    final response = await http.get(uri);
    return jsonDecode(response.body);
  }

  void _connect() async {
    if (!mounted || _isConnecting) return;

    setState(() {
      _isConnecting = true;
    });

    try {
      _channel = WebSocketChannel.connect(
        Uri.parse(dotenv.env['WS_STREAMING']!),
      );

      await _channel.ready;

      if (mounted) {
        setState(() {
          _isConnected = true;
          _isConnecting = false;
        });
      }

      _subscription = _channel.stream.listen(
        (message) {
          try {
            final data = jsonDecode(message);
            debugPrint("Tipo messaggio: ${data['type']}");
            if (data['type'] == 'update') {
              SensorUpdate sensorUpdate = SensorUpdate.fromJson(
                data['sensor_update'],
              );
              CalculatedIndex index = CalculatedIndex.fromJson(data['index']);
              Pattern pattern = Pattern.fromJson(data['pattern']);
              setState(() {
                allPatients[sensorUpdate.patientId] = Record(
                  sensorUpdate: sensorUpdate,
                  index: index,
                  pattern: pattern,
                );
                _lastUpdate = sensorUpdate;
              });

              Trend trend = Trend.fromJson(data['trend_update']);

              setState(() {
                allTrends[trend.patientId] = allTrends[trend.patientId] ?? [];
                allTrends[trend.patientId]!.add(trend);
                DateTime actualTime = DateTime.parse(trend.start);
                allTrends[trend.patientId]?.removeWhere((t) {
                  DateTime startTime = DateTime.parse(t.start);
                  return actualTime.difference(startTime).inMinutes > 5;
                });
              });
              _saveDataToLocal();
            } else if (data['type'] == 'ai_mex') {
              final int pid = data['patient_id'];
              final String full = data['text'] ?? '';

              // Cancella eventuale animazione precedente
              _aiTypingTimer?.cancel();
              _aiTypingIndex = 0;
              _aiFullText = full;

              setState(() {
                _aiStreams[pid] = StringBuffer();
                _aiResponses.remove(pid);
                _aiStreaming = true;
                _showAiPanel = true;
              });

              // Simula typing locale
              _aiTypingTimer = Timer.periodic(
                const Duration(milliseconds: 15),
                (_) {
                  if (_aiTypingIndex < _aiFullText.length) {
                    setState(() {
                      _aiStreams[pid]!.write(_aiFullText[_aiTypingIndex]);
                      _aiTypingIndex++;
                    });
                  } else {
                    _aiTypingTimer?.cancel();
                    _aiTypingTimer = null;
                    setState(() {
                      _aiResponses[pid] = _aiFullText;
                      _aiStreams[pid]?.clear();
                      _aiStreaming = false;
                    });
                  }
                },
              );
            }
          } catch (e) {
            debugPrint("Parsing error: $e");
          }
        },
        onDone: () {
          debugPrint("WebSocket chiuso dal server");
          if (mounted) _handleRetry();
        },
        onError: (error) {
          debugPrint("Errore WebSocket: $error");
          if (mounted) _handleRetry();
        },
      );
    } catch (e) {
      debugPrint("Errore di connessione: $e");
      _handleRetry();
    }
  }

  void _handleRetry() {
    if (!mounted) return;

    setState(() {
      _isConnected = false;
      _isConnecting = false;
    });

    // Annulla eventuali timer precedenti e ne avvia uno nuovo
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 5), () {
      debugPrint("Tentativo di riconnessione in corso...");
      if (mounted) _connect();
    });
  }

  Future<void> _saveDataToLocal() async {
    final prefs = await SharedPreferences.getInstance();
    // Converte la mappa dei pazienti in JSON string
    final patientsData = allPatients.map(
      (key, record) => MapEntry(key.toString(), {
        'sensorUpdate': record.sensorUpdate,
        'index': record.index,
        'pattern': record.pattern,
      }),
    );
    await prefs.setString('cached_patients', jsonEncode(patientsData));
  }

  Future<void> _loadDataFromLocal() async {
    final prefs = await SharedPreferences.getInstance();
    final String? cachedData = prefs.getString('cached_patients');

    if (cachedData != null) {
      final Map<String, dynamic> decoded = jsonDecode(cachedData);
      setState(() {
        decoded.forEach((key, value) {
          final int pid = int.parse(key);
          allPatients[pid] = Record(
            sensorUpdate: SensorUpdate.fromJson(value['sensorUpdate']),
            index: CalculatedIndex.fromJson(value['index']),
            pattern: Pattern.fromJson(value['pattern']),
          );
        });
      });
    }
  }

  @override
  void initState() {
    super.initState();
    debugPrint("Inizializzazione LivestreamPage...");
    _loadDataFromLocal();
    _connect();
  }

  @override
  void dispose() {
    _aiTypingTimer?.cancel();
    _subscription?.cancel();
    _channel.sink.close();
    _reconnectTimer?.cancel();
    super.dispose();
  }

  Widget _buildEmptyChartPlaceholder() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.show_chart,
            size: 60,
            color: Colors.white.withOpacity(0.05),
          ),
          const SizedBox(height: 16),
          Text(
            "In attesa di selezione...",
            style: TextStyle(
              color: Colors.white.withOpacity(0.2),
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChart() {
    final selectedPatient = allPatients[selectedPatientId]?.sensorUpdate;

    return Container(
      margin: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.05)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.4),
            blurRadius: 15,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        children: [
          // 1. HEADER DEL GRAFICO
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "LIVE TREND ANALYSIS",
                      style: TextStyle(
                        color: chartAttributes
                            .firstWhere(
                              (attr) => attr['label'] == label,
                            )['color']
                            .withOpacity(0.8),
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.5,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      selectedPatient != null
                          ? "Patient: ${selectedPatient.patientId}"
                          : "Seleziona un paziente",
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                if (selectedPatient != null)
                  _buildLiveBadge((() {
                    switch (label) {
                      case "Heart Rate":
                        return selectedPatient.heartRate;
                      case "SpO2":
                        return selectedPatient.oxygenSaturation;
                      case "Temperature":
                        return selectedPatient.bodyTemperature;
                      case "Derived_MAP":
                        return selectedPatient.derivedMap;
                      default:
                        return selectedPatient.respiratoryRate;
                    }
                  }())),
              ],
            ),
          ),
          if (selectedPatient != null)
            IconButtonRow(
              onPressed: (label) {
                setState(() {
                  this.label = label;
                });
              },
              isSelected: (label) => this.label == label,
            ),
          const Divider(height: 1, color: Colors.white10),
          // 2. AREA DEL GRAFICO
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(15, 20, 20, 10),
              child: selectedPatient != null
                  ? PatientTrendChart(
                      history: allTrends[selectedPatientId] ?? [],
                      lineColor: chartAttributes
                          .firstWhere((attr) => attr['label'] == label)['color']
                          .withOpacity(0.8),
                      label: label,
                    )
                  : _buildEmptyChartPlaceholder(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAiPanel() {
    return AiExplanationPanel(
      patientId: selectedPatientId,
      streamingText: _aiStreams[selectedPatientId]?.toString() ?? '',
      completedText: _aiResponses[selectedPatientId],
      isStreaming: _aiStreaming,
    );
  }

  // Widget per il piccolo badge pulsante con il valore attuale
  Widget _buildLiveBadge(num value) {
    Color col = chartAttributes
        .firstWhere((attr) => attr['label'] == label)['color']
        .withOpacity(0.8);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: col.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: col),
      ),
      child: Row(
        children: [
          Icon(Icons.sensors, color: col, size: 16),
          const SizedBox(width: 8),
          Text(
            "${value.toStringAsFixed(2)} $label",
            style: TextStyle(
              color: col,
              fontWeight: FontWeight.bold,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
    );
  }

  dynamic addRiskCategory() {
    String riskCategory =
        allPatients[selectedPatientId]!.sensorUpdate.prediction;
    return {
      "Risk Category": riskCategory,
      "Avg_ShockIndex": allPatients[selectedPatientId]!.index.shockIndex,
      "Avg_ModifiedShockIndex":
          allPatients[selectedPatientId]!.index.modifiedShockIndex,
      "Avg_AgeShockIndex_Norm":
          allPatients[selectedPatientId]!.index.ageIndex / 100,
      "Avg_DiastolicShockIndex":
          allPatients[selectedPatientId]!.index.diastolicShockIndex,
      "Avg_PulsePressureIndex": allPatients[selectedPatientId]!.index.ppIndex,
    };
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // --- RIGA SUPERIORE (Quadrante 1 e 2) ---
            Expanded(
              flex: 5,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 1° Quadrante: Classificazione Live / AI Panel (Sinistra)
                  Expanded(
                    flex: 1,
                    child: Column(
                      children: [
                        // Rende il pannello flessibile così riempie lo spazio sopra il bottone
                        Expanded(
                          child: buildGlassPanel(child: _buildAiPanel()),
                        ),
                        if (selectedPatientId != null) ...[
                          const SizedBox(height: 8),
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 12),
                            child: ElevatedButton.icon(
                              onPressed: () async {
                                setState(() {
                                  _aiStreaming = true;
                                  _showAiPanel = true;
                                });

                                final data = await _requestAiExplanation(
                                  selectedPatientId!,
                                );
                                final int pid = data['patient_id'];
                                final String full = data['message'] ?? '';

                                _aiTypingTimer?.cancel();
                                _aiTypingIndex = 0;
                                _aiFullText = full;

                                setState(() {
                                  _aiStreams[pid] = StringBuffer();
                                  _aiResponses.remove(pid);
                                });

                                _aiTypingTimer = Timer.periodic(
                                  const Duration(milliseconds: 15),
                                  (_) {
                                    if (_aiTypingIndex < _aiFullText.length) {
                                      setState(() {
                                        _aiStreams[pid]!.write(
                                          _aiFullText[_aiTypingIndex],
                                        );
                                        _aiTypingIndex++;
                                      });
                                    } else {
                                      _aiTypingTimer?.cancel();
                                      _aiTypingTimer = null;
                                      setState(() {
                                        _aiResponses[pid] = _aiFullText;
                                        _aiStreams[pid]?.clear();
                                        _aiStreaming = false;
                                      });
                                    }
                                  },
                                );
                              },
                              icon: const Icon(Icons.auto_awesome, size: 14),
                              label: const Text("Explain with AI"),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(
                                  0xFF7F77DD,
                                ).withOpacity(0.15),
                                foregroundColor: const Color(0xFFAFA9EC),
                                side: const BorderSide(
                                  color: Color(0xFF7F77DD),
                                  width: 0.8,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10),
                                ),
                                elevation: 0,
                                minimumSize: const Size.fromHeight(
                                  45,
                                ), // Forza il bottone ad essere ben visibile
                              ),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(width: 16),

                  // 2° Quadrante: Dettagli, Gauge e Badge (Destra)
                  Expanded(
                    flex: 1,
                    child: Column(
                      children: [
                        // Sotto-riga 1: Quick Status & Emodynamic Status
                        Expanded(
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Expanded(
                                child: buildGlassPanel(
                                  child: Column(
                                    children: [
                                      const Text(
                                        "QUICK STATUS",
                                        style: TextStyle(
                                          color: Colors.white54,
                                          fontSize: 10,
                                          letterSpacing: 1.2,
                                        ),
                                      ),
                                      const Divider(color: Colors.white10),
                                      Expanded(
                                        child:
                                            selectedPatientId != null &&
                                                allTrends[selectedPatientId]
                                                        ?.isNotEmpty ==
                                                    true
                                            ? CircularSummaryPanel(
                                                trend:
                                                    allTrends[selectedPatientId]!
                                                        .last,
                                                rpp:
                                                    allPatients[selectedPatientId]!
                                                        .index
                                                        .ratePp,
                                                pp: allPatients[selectedPatientId]!
                                                    .index
                                                    .ppIndex,
                                              )
                                            : _buildSimplePlaceholder(
                                                "Seleziona paziente",
                                              ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: buildGlassPanel(
                                  child: Column(
                                    children: [
                                      const Text(
                                        "EMODYNAMIC STATUS",
                                        style: TextStyle(
                                          color: Colors.white54,
                                          fontSize: 10,
                                          letterSpacing: 1.2,
                                        ),
                                      ),
                                      const Divider(color: Colors.white10),
                                      // --- FIX: il radar veniva tagliato in alto (vertice "SI")
                                      // perché AspectRatio calcolava l'altezza in base alla
                                      // larghezza disponibile (spesso maggiore dello spazio
                                      // verticale reale del pannello), e fl_chart disegna le
                                      // etichette dei vertici leggermente oltre il bordo del
                                      // poligono. Ora vincoliamo il radar a un quadrato basato
                                      // sul lato PIÙ CORTO disponibile, con un margine
                                      // riservato esplicitamente alle etichette: così l'intero
                                      // grafico rientra sempre nella viewport, senza bisogno
                                      // di scroll.
                                      selectedPatientId != null
                                          ? Expanded(
                                              child: LayoutBuilder(
                                                builder: (context, constraints) {
                                                  const double labelPadding =
                                                      28.0;
                                                  final double side =
                                                      (constraints.maxWidth <
                                                              constraints
                                                                  .maxHeight
                                                          ? constraints.maxWidth
                                                          : constraints
                                                                .maxHeight) -
                                                      labelPadding;

                                                  return Center(
                                                    child: Padding(
                                                      padding:
                                                          const EdgeInsets.symmetric(
                                                            vertical:
                                                                labelPadding /
                                                                2,
                                                          ),
                                                      child: SizedBox(
                                                        width: side > 0
                                                            ? side
                                                            : 0,
                                                        height: side > 0
                                                            ? side
                                                            : 0,
                                                        child: ClinicalRadarChart(
                                                          radarData: [
                                                            addRiskCategory(),
                                                          ],
                                                        ),
                                                      ),
                                                    ),
                                                  );
                                                },
                                              ),
                                            )
                                          : _buildSimplePlaceholder(
                                              "Seleziona paziente",
                                            ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                        // Sotto-riga 2: Trend Insights & Risk Indicators
                        Expanded(
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Expanded(
                                child: buildGlassPanel(
                                  child: Column(
                                    children: [
                                      const Text(
                                        "TREND INSIGHTS",
                                        style: TextStyle(
                                          color: Colors.white54,
                                          fontSize: 10,
                                          letterSpacing: 1.2,
                                        ),
                                      ),
                                      const Divider(color: Colors.white10),
                                      Expanded(
                                        child: SingleChildScrollView(
                                          physics:
                                              const BouncingScrollPhysics(),
                                          child: FeaturePanel(
                                            insights:
                                                (selectedPatientId != null &&
                                                    allTrends[selectedPatientId]
                                                            ?.isNotEmpty ==
                                                        true)
                                                ? getFeatureInsights(
                                                    allTrends[selectedPatientId]!
                                                        .last,
                                                  )
                                                : [],
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: _buildRiskBadgesPanel({
                                  "hemo_deterioration":
                                      selectedPatientId == null
                                      ? false
                                      : allPatients[selectedPatientId]
                                            ?.pattern
                                            .hemoDeterioration,
                                  "resp_failure": selectedPatientId == null
                                      ? false
                                      : allPatients[selectedPatientId]
                                            ?.pattern
                                            .progRespFailure,
                                  "dynamic_sepsis": selectedPatientId == null
                                      ? false
                                      : allPatients[selectedPatientId]
                                            ?.pattern
                                            .dynamicSepsis,
                                }),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // --- RIGA INFERIORE (Quadrante 3 e 4) ---
            Expanded(
              flex: 4,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 3° Quadrante: Lista Triage (Sinistra)
                  Expanded(
                    flex: 1,
                    child: buildGlassPanel(
                      child: TriageMasterView(
                        allPatients: allPatients.map(
                          (key, value) => MapEntry(key, value.sensorUpdate),
                        ),
                        selectedPatientId: selectedPatientId,
                        onPatientSelected: (p) =>
                            setState(() => selectedPatientId = p.patientId),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  // 4° Quadrante: Grafico
                  Expanded(
                    flex: 1,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [Expanded(child: _buildChart())],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Helper per i placeholder quando non c'è selezione
  Widget _buildSimplePlaceholder(String text) {
    return Center(
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white10,
          fontSize: 12,
          fontStyle: FontStyle.italic,
        ),
      ),
    );
  }
}
