import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:frontend_clinic_risk/livestream_page.dart';
import 'widget/sidebar.dart';
import 'history_page.dart';
import 'analytics.dart';
import 'evaluation.dart';

Future<void> main() async {
  await dotenv.load(fileName: ".env");
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Clinic Risk Analysis',
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
  int selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;
    bool useCollapsed = screenWidth < 1000;
    bool isMobile = screenWidth < 600;

    Widget buildMainPage() {
      if (selectedIndex == 0) {
        return LivestreamPage();
      } else if (selectedIndex == 1) {
        return HistoryPage();
      } else if (selectedIndex == 2) {
        return AnalyticsPage();
      } else if (selectedIndex == 3) {
        return EvaluationPage();
      }
      return const Center();
    }

    return Scaffold(
      backgroundColor: const Color(0xFF151515),
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
