import 'dart:async';
import 'package:flutter/material.dart' hide Chip;
import 'package:frontend_clinic_risk/types/indexclass.dart';
import 'package:frontend_clinic_risk/widget/feature.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';
import 'widget/classification_panel.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'widget/patient_card.dart';
import 'types/trend.dart';
import 'types/sensorUpdate.dart';
import 'types/pattern.dart';
import 'widget/trend_graph.dart';
import 'widget/iconbuttonrow.dart';

class LivestreamPage extends StatefulWidget {
  const LivestreamPage({super.key});

  @override
  State<LivestreamPage> createState() => _LivestreamPageState();
}

class Record {
  SensorUpdate sensorUpdate;
  Index index;
  Pattern pattern;

  Record({
    required this.sensorUpdate,
    required this.index,
    required this.pattern,
  });
}

class _LivestreamPageState extends State<LivestreamPage> {
  late WebSocketChannel _channel;
  Timer? _reconnectTimer;
  bool _isConnected = false;
  bool _isConnecting = false;
  Map<int, Record> allPatients = {};
  Map<int, List<Trend>> allTrends = {};
  SensorUpdate? _lastUpdate;
  int? selectedPatientId;
  String label = "Heart Rate";

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

      // Fondamentale: ascoltiamo lo stream qui per gestire la disconnessione
      _channel.stream.listen(
        (message) {
          try {
            final data = jsonDecode(message);
            SensorUpdate sensorUpdate = SensorUpdate.fromJson(
              data['sensor_update'],
            );
            Index index = Index.fromJson(data['index_update']);
            Pattern pattern = Pattern.fromJson(data['pattern_update']);
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
          } catch (e) {
            debugPrint("Parsing error: $e");
          }
        },
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

  void _handleRetry() {
    if (mounted) {
      setState(() {
        _isConnected = false;
        _isConnecting = false;
      });

      // Annulla eventuali timer precedenti e ne avvia uno nuovo
      _reconnectTimer?.cancel();
      _reconnectTimer = Timer(const Duration(seconds: 5), () {
        debugPrint("Tentativo di riconnessione in corso...");
        _connect();
      });
    }
  }

  @override
  void initState() {
    super.initState();
    debugPrint("Inizializzazione LivestreamPage...");
    _connect();
  }

  @override
  void dispose() {
    _channel.sink.close();
    super.dispose();
  }

  Widget _buildStreamPanel() {
    return Center(
      child: _isConnected && _lastUpdate != null
          ? LiveClassificationPane(
              sensorUpdate: _lastUpdate!,
              isConnected: _isConnected,
            )
          : LiveClassificationPane(
              sensorUpdate: SensorUpdate(
                patientId: 0,
                heartRate: 0,
                respiratoryRate: 0,
                timestamp: '',
                bodyTemperature: 0,
                oxygenSaturation: 0,
                systolicBloodPressure: 0,
                diastolicBloodPressure: 0,
                age: 0,
                gender: '',
                weight: 0,
                height: 0,
                derivedHrv: 0,
                derivedPulsePressure: 0,
                derivedBmi: 0,
                derivedMap: 0,
                prediction: "",
              ),
              isConnected: _isConnected ? false : _isConnected,
            ),
    );
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
                    //if (selectedPatient != null) _buildHeaderChips(),
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

  Widget _buildStatCard(String label, double value, String unit, Color color) {
    return Column(
      children: [
        Stack(
          alignment: Alignment.center,
          children: [
            SizedBox(
              width: 60,
              height: 60,
              child: CircularProgressIndicator(
                value: value / 100, // Es: 0.98 per 98%
                strokeWidth: 6,
                backgroundColor: Colors.white10,
                valueColor: AlwaysStoppedAnimation<Color>(color),
              ),
            ),
            Text(
              value.toStringAsFixed(0),
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: Colors.white,
              ),
            ),
          ],
        ),
        SizedBox(height: 8),
        Text(label, style: TextStyle(fontSize: 10, color: Colors.white54)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        color: const Color.fromARGB(255, 46, 46, 46),
        child: Column(
          // Organizziamo lo schermo in due grandi righe (Top e Bottom)
          children: [
            // RIGA SUPERIORE (Quadranti 1 e 2)
            Expanded(
              flex: 1, // Metà dell'altezza totale
              child: Row(
                children: [
                  // 1° Quadrante: Stream Panel
                  Expanded(child: _buildStreamPanel()),
                  // 2° Quadrante: Placeholder (non ancora definito)
                  Expanded(
                    child: Column(
                      children: [
                        _buildStatCard(
                          "SpO₂",
                          allPatients[selectedPatientId]
                                  ?.sensorUpdate
                                  .oxygenSaturation ??
                              0,
                          "%",
                          chartAttributes[1]['color'],
                        ),
                        FeaturePanel(
                          insights:
                              (selectedPatientId != null &&
                                  allTrends[selectedPatientId]?.isNotEmpty ==
                                      true)
                              ? [
                                  FeatureInsight(
                                    name: "Heart Rate Δ%",
                                    deltaPercentage:
                                        allTrends[selectedPatientId]!
                                            .last
                                            .hrPct,
                                    value: allTrends[selectedPatientId]!
                                        .last
                                        .hrPct,
                                  ),
                                ]
                              : [],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // RIGA INFERIORE (Quadranti 3 e 4)
            Expanded(
              flex: 1, // L'altra metà dell'altezza
              child: Row(
                children: [
                  // 3° Quadrante: Pannello con il grafico
                  Expanded(child: _buildChart()),
                  Expanded(
                    child: TriageMasterView(
                      allPatients: allPatients.map(
                        (key, value) => MapEntry(key, value.sensorUpdate),
                      ),
                      selectedPatientId: selectedPatientId,
                      onPatientSelected: (p) {
                        setState(() {
                          selectedPatientId = p.patientId;
                        });
                      },
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
}
