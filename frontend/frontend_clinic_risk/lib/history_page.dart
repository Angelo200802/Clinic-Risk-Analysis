import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  final String url = Uri.parse(dotenv.env["BACKEND_BASE_API"]!).toString();

  @override
  Widget build(BuildContext context) {
    return const Center(child: Text("Patient History"));
  }
}
