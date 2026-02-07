import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';
import 'widget/classification_panel.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'widget/patient_card.dart';

class LivestreamPage extends StatefulWidget {
  const LivestreamPage({super.key});

  @override
  State<LivestreamPage> createState() => _LivestreamPageState();
}

class _LivestreamPageState extends State<LivestreamPage> {
  late WebSocketChannel _channel;
  bool _isConnected = false;
  bool _isConnecting = true;
  List<SensorUpdate> allPatients = [];
  int? selectedPatientId;

  void _connect() async {
    try {
      _channel = WebSocketChannel.connect(
        Uri.parse(dotenv.env['WS_STREAMING']!),
      );

      // Aspetta che il protocollo WebSocket sia effettivamente stabilito
      await _channel.ready;

      setState(() {
        _isConnected = true;
        _isConnecting = false;
      });
    } catch (e) {
      debugPrint("Errore di connessione: $e");
      setState(() {
        _isConnected = false;
        _isConnecting = false;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _connect();
  }

  @override
  void dispose() {
    _channel.sink.close();
    super.dispose();
  }

  Widget _buildStreamPanel() {
    return Center(
      child: _isConnected
          ? StreamBuilder(
              stream: _channel.stream,
              builder: (context, snapshot) {
                if (snapshot.hasError) {
                  return Text(
                    "Errore: ${snapshot.error}",
                    style: TextStyle(color: Colors.red),
                  );
                }
                if (!snapshot.hasData) {
                  return const CircularProgressIndicator();
                }
                if (snapshot.hasData) {
                  try {
                    final data = jsonDecode(snapshot.data!);
                    final sensorUpdate = SensorUpdate.fromJson(data['data']);
                    allPatients.add(sensorUpdate);
                    return LiveClassificationPane(
                      sensorUpdate: sensorUpdate,
                      isConnected: _isConnected,
                    );
                  } catch (e) {
                    return Text('Error parsing data: $e');
                  }
                } else if (snapshot.hasError) {
                  return Text('WebSocket error: ${snapshot.error}');
                } else {
                  return const CircularProgressIndicator();
                }
              },
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
              isConnected: _isConnected,
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
