import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'widget/classification_panel.dart';
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
  final _channel = WebSocketChannel.connect(
    Uri.parse(dotenv.env['WS_STREAMING']!),
  );

  @override
  void dispose() {
    _channel.sink.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Patient Risk Dashboard')),
      body: Center(
        child: StreamBuilder(
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
                return LiveClassificationPane(sensorUpdate: sensorUpdate);
              } catch (e) {
                return Text('Error parsing data: $e');
              }
            } else if (snapshot.hasError) {
              return Text('WebSocket error: ${snapshot.error}');
            } else {
              return const CircularProgressIndicator();
            }
          },
        ),
      ),
    );
  }
}
