import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'widget/classification_panel.dart';
import 'widget/sidebar.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

Future<void> main() async {
  await dotenv.load(fileName: ".env");
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: DashboardPage(),
    );
  }
}

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  late WebSocketChannel _channel;
  bool _isConnected = false;
  bool _isConnecting = true;

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

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;
    bool useCollapsed = screenWidth < 1000;
    bool isMobile = screenWidth < 600;
    int selectedIndex = 0;

    Widget buildMainPage() {
      if (selectedIndex == 0) {
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
                        final sensorUpdate = SensorUpdate.fromJson(data);
                        debugPrint("DEBUG KEYS: ${data.keys.toList()}");
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
      return const Center();
    }

    return Scaffold(
      backgroundColor: const Color.fromARGB(255, 46, 46, 46),
      drawer: isMobile
          ? Drawer(
              child: SidebarComponent(
                selectedIndex: selectedIndex,
                onItemSelected: (index) {
                  setState(() => selectedIndex = index);
                  Navigator.pop(context); // Chiude il drawer dopo la selezione,
                },
                isCollapsed: useCollapsed,
              ),
            )
          : null,
      appBar: isMobile ? AppBar(backgroundColor: Colors.transparent) : null,
      body: Row(
        children: [
          SidebarComponent(
            selectedIndex: selectedIndex,
            onItemSelected: (index) {
              setState(() => selectedIndex = index);
            },
            isCollapsed: useCollapsed,
          ),
          Expanded(
            child: Container(
              color: const Color.fromARGB(255, 46, 46, 46),
              child: buildMainPage(),
            ),
          ),
        ],
      ),
    );
  }
}
