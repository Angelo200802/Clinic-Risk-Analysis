import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class InsigthPage extends StatefulWidget {
  const InsigthPage({super.key});

  @override
  State<InsigthPage> createState() => _InsigthPageState();
}

class _InsigthPageState extends State<InsigthPage> {
  final String url = Uri.parse(dotenv.env["BACKEND_BASE_API"]!).toString();

  @override
  Widget build(BuildContext context) {
    return const Center(child: Text("Patient History"));
  }
}
