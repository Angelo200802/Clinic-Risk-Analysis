import 'dart:async';

import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';
import 'widget/classification_panel.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'widget/patient_card.dart';

class Trend {}

class LivestreamPage extends StatefulWidget {
  const LivestreamPage({super.key});

  @override
  State<LivestreamPage> createState() => _LivestreamPageState();
}

class _LivestreamPageState extends State<LivestreamPage> {
  late WebSocketChannel _channel;
  Timer? _reconnectTimer;
  bool _isConnected = false;
  bool _isConnecting = false;
  Map<int, SensorUpdate> allPatients = {};
  Map<int, Trend> allTrends = {};
  SensorUpdate? _lastUpdate;
  int? selectedPatientId;

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
      _channel.stream.listen((message) {
        try {
          final data = jsonDecode(message);
          if (data['type'] == 'prediction') {
            SensorUpdate sensorUpdate = SensorUpdate.fromJson(data['data']);
            setState(() {
              allPatients[sensorUpdate.patientId] = sensorUpdate;
              _lastUpdate = sensorUpdate;
            });
          } else {
            debugPrint(
              "Messaggio di tipo sconosciuto ricevuto: \n${data['data']}",
            );
          }
        } catch (e) {
          debugPrint("Parsing error: $e");
        }
      });
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        // Il Container alla riga 81 del tuo main.dart
        color: const Color.fromARGB(255, 46, 46, 46),
        child: Row(
          // Il Row alla riga 71
          children: [
            // 1. IL PANNELLO STREAM DEVE AVERE UN LIMITE
            // Se lo stream_panel contiene una lista, avvolgilo in Expanded o SizedBox
            Expanded(
              flex: 2, // Prende 2 parti dello spazio
              child: _buildStreamPanel(),
            ),

            // 2. IL TRIAGE MASTER DEVE AVERE UN LIMITE
            Expanded(
              flex: 3, // Prende 3 parti dello spazio
              child: TriageMasterView(
                allPatients: allPatients,
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
    );
  }
}
