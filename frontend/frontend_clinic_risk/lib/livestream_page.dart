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

class LivestreamDataService extends ChangeNotifier {
  LivestreamDataService._internal();
  static final LivestreamDataService instance =
      LivestreamDataService._internal();

  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  Timer? _reconnectTimer;
  bool _initialized = false;

  bool isConnected = false;
  bool isConnecting = false;

  Map<int, Record> allPatients = {};
  Map<int, List<Trend>> allTrends = {};
  // JSON grezzo dei trend, usato solo per la persistenza su disco
  final Map<int, List<Map<String, dynamic>>> _rawTrends = {};

  int? selectedPatientId;

  final Map<int, StringBuffer> aiStreams = {};
  final Map<int, String> aiResponses = {};
  bool aiStreaming = false;
  Timer? _aiTypingTimer;
  int _aiTypingIndex = 0;
  String _aiFullText = '';

  /// Idempotente: se il servizio è già connesso/inizializzato, non fa nulla.
  /// Questo è il punto chiave che risolve il problema: chiamarlo di nuovo
  /// quando l'utente torna sulla pagina NON riavvia la connessione né
  /// azzera i dati già raccolti.
  Future<void> init() async {
    if (_initialized) return;
    _initialized = true;
    await _loadDataFromLocal();
    _connect();
  }

  void selectPatient(int? id) {
    selectedPatientId = id;
    notifyListeners();
  }

  Future<Map<String, dynamic>> requestAiExplanation(int patientId) async {
    // Spostato qui solo per coerenza; se preferisci lascialo nella pagina.
    final uri = Uri.parse(
      'http://clinc-risk-analysis-1:8081/explain/$patientId',
    );
    final response = await http.get(uri);
    return jsonDecode(response.body);
  }

  void _connect() async {
    if (isConnecting) return;
    isConnecting = true;
    notifyListeners();

    try {
      _channel = WebSocketChannel.connect(
        Uri.parse(dotenv.env['WS_STREAMING']!),
      );
      await _channel!.ready;

      isConnected = true;
      isConnecting = false;
      notifyListeners();

      _subscription = _channel!.stream.listen(
        _onMessage,
        onDone: () {
          debugPrint("WebSocket chiuso dal server");
          _handleRetry();
        },
        onError: (error) {
          debugPrint("Errore WebSocket: $error");
          _handleRetry();
        },
      );
    } catch (e) {
      debugPrint("Errore di connessione: $e");
      _handleRetry();
    }
  }

  void _onMessage(dynamic message) {
    try {
      final data = jsonDecode(message);
      if (data['type'] == 'update') {
        final sensorUpdate = SensorUpdate.fromJson(data['sensor_update']);
        final index = CalculatedIndex.fromJson(data['index']);
        final pattern = Pattern.fromJson(data['pattern']);

        allPatients[sensorUpdate.patientId] = Record(
          sensorUpdate: sensorUpdate,
          index: index,
          pattern: pattern,
        );

        final trendJson = data['trend_update'] as Map<String, dynamic>;
        final trend = Trend.fromJson(trendJson);
        final pid = trend.patientId;

        allTrends[pid] = allTrends[pid] ?? [];
        allTrends[pid]!.add(trend);
        _rawTrends[pid] = _rawTrends[pid] ?? [];
        _rawTrends[pid]!.add(trendJson);

        final actualTime = DateTime.parse(trend.start);
        bool isOld(String start) =>
            actualTime.difference(DateTime.parse(start)).inMinutes > 5;

        allTrends[pid]?.removeWhere((t) => isOld(t.start));
        _rawTrends[pid]?.removeWhere((t) => isOld(t['start']));

        notifyListeners();
        _saveDataToLocal();
      } else if (data['type'] == 'ai_mex') {
        _handleAiMessage(data);
      }
    } catch (e) {
      debugPrint("Parsing error: $e");
    }
  }

  void _handleAiMessage(Map<String, dynamic> data) {
    final int pid = data['patient_id'];
    final String full = data['text'] ?? '';

    _aiTypingTimer?.cancel();
    _aiTypingIndex = 0;
    _aiFullText = full;

    aiStreams[pid] = StringBuffer();
    aiResponses.remove(pid);
    aiStreaming = true;
    notifyListeners();

    _aiTypingTimer = Timer.periodic(const Duration(milliseconds: 15), (_) {
      if (_aiTypingIndex < _aiFullText.length) {
        aiStreams[pid]!.write(_aiFullText[_aiTypingIndex]);
        _aiTypingIndex++;
        notifyListeners();
      } else {
        _aiTypingTimer?.cancel();
        _aiTypingTimer = null;
        aiResponses[pid] = _aiFullText;
        aiStreams[pid]?.clear();
        aiStreaming = false;
        notifyListeners();
      }
    });
  }

  void _handleRetry() {
    isConnected = false;
    isConnecting = false;
    notifyListeners();

    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 5), _connect);
  }

  Future<void> _saveDataToLocal() async {
    final prefs = await SharedPreferences.getInstance();

    final patientsData = allPatients.map(
      (key, record) => MapEntry(key.toString(), {
        'sensorUpdate': record.sensorUpdate,
        'index': record.index,
        'pattern': record.pattern,
      }),
    );
    await prefs.setString('cached_patients', jsonEncode(patientsData));

    final trendsData = _rawTrends.map((k, v) => MapEntry(k.toString(), v));
    await prefs.setString('cached_trends', jsonEncode(trendsData));
  }

  Future<void> _loadDataFromLocal() async {
    final prefs = await SharedPreferences.getInstance();

    final String? cachedPatients = prefs.getString('cached_patients');
    if (cachedPatients != null) {
      final Map<String, dynamic> decoded = jsonDecode(cachedPatients);
      decoded.forEach((key, value) {
        final int pid = int.parse(key);
        allPatients[pid] = Record(
          sensorUpdate: SensorUpdate.fromJson(value['sensorUpdate']),
          index: CalculatedIndex.fromJson(value['index']),
          pattern: Pattern.fromJson(value['pattern']),
        );
      });
    }

    final String? cachedTrends = prefs.getString('cached_trends');
    if (cachedTrends != null) {
      final Map<String, dynamic> decoded = jsonDecode(cachedTrends);
      decoded.forEach((key, value) {
        final int pid = int.parse(key);
        final list = (value as List).cast<Map<String, dynamic>>();
        _rawTrends[pid] = list;
        allTrends[pid] = list.map((t) => Trend.fromJson(t)).toList();
      });
    }

    notifyListeners();
  }

  /// Da chiamare SOLO alla chiusura reale dell'app (es. in un listener
  /// di lifecycle globale), mai dal dispose() di una singola pagina.
  void shutdown() {
    _aiTypingTimer?.cancel();
    _subscription?.cancel();
    _channel?.sink.close();
    _reconnectTimer?.cancel();
  }
}

class _LivestreamPageState extends State<LivestreamPage> {
  final LivestreamDataService _service = LivestreamDataService.instance;

  bool _showAiPanel = false; // resta locale: è solo UI della pagina, non dato
  String label = "Heart Rate";
  @override
  void initState() {
    super.initState();
    debugPrint("Inizializzazione LivestreamPage...");
    _service.init(); // no-op se già connesso: niente reset
    _service.addListener(_onServiceUpdate);
  }

  void _onServiceUpdate() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _service.removeListener(_onServiceUpdate);
    // NIENTE _channel.sink.close() qui: il websocket deve restare
    // vivo anche quando l'utente lascia la pagina.
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
    final selectedPatient =
        _service.allPatients[_service.selectedPatientId]?.sensorUpdate;

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
                      history:
                          _service.allTrends[_service.selectedPatientId] ?? [],
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
      patientId: _service.selectedPatientId,
      streamingText:
          _service.aiStreams[_service.selectedPatientId]?.toString() ?? '',
      completedText: _service.aiResponses[_service.selectedPatientId],
      isStreaming: _service.aiStreaming,
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
    String riskCategory = _service
        .allPatients[_service.selectedPatientId]!
        .sensorUpdate
        .prediction;
    return {
      "Risk Category": riskCategory,
      "Avg_ShockIndex":
          _service.allPatients[_service.selectedPatientId]!.index.shockIndex,
      "Avg_ModifiedShockIndex": _service
          .allPatients[_service.selectedPatientId]!
          .index
          .modifiedShockIndex,
      "Avg_AgeShockIndex_Norm":
          _service.allPatients[_service.selectedPatientId]!.index.ageIndex /
          100,
      "Avg_DiastolicShockIndex": _service
          .allPatients[_service.selectedPatientId]!
          .index
          .diastolicShockIndex,
      "Avg_PulsePressureIndex":
          _service.allPatients[_service.selectedPatientId]!.index.ppIndex,
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
                        if (_service.selectedPatientId != null) ...[
                          const SizedBox(height: 8),
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 12),
                            child: ElevatedButton.icon(
                              onPressed: () async {
                                setState(() {
                                  _service.aiStreaming = true;
                                  _showAiPanel = true;
                                });

                                final data = await _service
                                    .requestAiExplanation(
                                      _service.selectedPatientId!,
                                    );
                                final int pid = data['patient_id'];
                                final String full = data['message'] ?? '';

                                _service._aiTypingTimer?.cancel();
                                _service._aiTypingIndex = 0;
                                _service._aiFullText = full;

                                setState(() {
                                  _service.aiStreams[pid] = StringBuffer();
                                  _service.aiResponses.remove(pid);
                                });

                                _service._aiTypingTimer = Timer.periodic(
                                  const Duration(milliseconds: 15),
                                  (_) {
                                    if (_service._aiTypingIndex <
                                        _service._aiFullText.length) {
                                      setState(() {
                                        _service.aiStreams[pid]!.write(
                                          _service._aiFullText[_service
                                              ._aiTypingIndex],
                                        );
                                        _service._aiTypingIndex++;
                                      });
                                    } else {
                                      _service._aiTypingTimer?.cancel();
                                      _service._aiTypingTimer = null;
                                      setState(() {
                                        _service.aiResponses[pid] =
                                            _service._aiFullText;
                                        _service.aiStreams[pid]?.clear();
                                        _service.aiStreaming = false;
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
                                            _service.selectedPatientId !=
                                                    null &&
                                                _service
                                                        .allTrends[_service
                                                            .selectedPatientId]
                                                        ?.isNotEmpty ==
                                                    true
                                            ? CircularSummaryPanel(
                                                trend: _service
                                                    .allTrends[_service
                                                        .selectedPatientId]!
                                                    .last,
                                                rpp: _service
                                                    .allPatients[_service
                                                        .selectedPatientId]!
                                                    .index
                                                    .ratePp,
                                                pp: _service
                                                    .allPatients[_service
                                                        .selectedPatientId]!
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
                                      _service.selectedPatientId != null
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
                                                (_service.selectedPatientId !=
                                                        null &&
                                                    _service
                                                            .allTrends[_service
                                                                .selectedPatientId]
                                                            ?.isNotEmpty ==
                                                        true)
                                                ? getFeatureInsights(
                                                    _service
                                                        .allTrends[_service
                                                            .selectedPatientId]!
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
                                      _service.selectedPatientId == null
                                      ? false
                                      : _service
                                            .allPatients[_service
                                                .selectedPatientId]
                                            ?.pattern
                                            .hemoDeterioration,
                                  "resp_failure":
                                      _service.selectedPatientId == null
                                      ? false
                                      : _service
                                            .allPatients[_service
                                                .selectedPatientId]
                                            ?.pattern
                                            .progRespFailure,
                                  "dynamic_sepsis":
                                      _service.selectedPatientId == null
                                      ? false
                                      : _service
                                            .allPatients[_service
                                                .selectedPatientId]
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
                        allPatients: _service.allPatients.map(
                          (key, value) => MapEntry(key, value.sensorUpdate),
                        ),
                        selectedPatientId: _service.selectedPatientId,
                        onPatientSelected: (p) => setState(
                          () => _service.selectedPatientId = p.patientId,
                        ),
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
