import 'package:flutter/material.dart';

class CardiacStressRanking extends StatelessWidget {
  final List<dynamic> data;

  const CardiacStressRanking({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    // Dividiamo i dati per genere per una visualizzazione comparativa
    final males = data.where((e) => e["Gender"] == "Male").toList();
    final females = data.where((e) => e["Gender"] == "Female").toList();

    return DefaultTabController(
      length: 2,
      child: Column(
        children: [
          const TabBar(
            tabs: [
              Tab(icon: Icon(Icons.male), text: "Uomini"),
              Tab(icon: Icon(Icons.female), text: "Donne"),
            ],
            indicatorColor: Colors.redAccent,
            labelColor: Colors.white,
          ),
          SizedBox(
            height: 400, // Altezza fissa per lo scroll interno
            child: TabBarView(
              children: [_buildStressList(males), _buildStressList(females)],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStressList(List<dynamic> patients) {
    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: patients.length,
      itemBuilder: (context, index) {
        final p = patients[index];
        final double rpp = p["Rate_Pressure_Product"].toDouble();

        // Calcolo di una percentuale fittizia per la barra di progresso (es. su max 20000)
        double progress = (rpp / 20000).clamp(0.0, 1.0);

        return Card(
          color: Colors.grey[900],
          margin: const EdgeInsets.only(bottom: 12),
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      "Paziente #${p['Patient ID']}",
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.redAccent.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        "Rank ${p['rank']}",
                        style: const TextStyle(
                          color: Colors.redAccent,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    const Icon(
                      Icons.favorite,
                      color: Colors.redAccent,
                      size: 18,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      "RPP: ${rpp.toInt()}",
                      style: const TextStyle(
                        fontSize: 20,
                        color: Colors.white,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                LinearProgressIndicator(
                  value: progress,
                  backgroundColor: Colors.white10,
                  color: Color.lerp(Colors.orange, Colors.red, progress),
                  minHeight: 6,
                ),
                const SizedBox(height: 4),
                Text(
                  "Età: ${p['Age']} - Sistolica: ${p['Systolic Blood Pressure']} mmHg",
                  style: const TextStyle(color: Colors.white54, fontSize: 12),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class ObesityMismatchList extends StatelessWidget {
  final List<dynamic> data;

  const ObesityMismatchList({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
          child: Text(
            "Anomalie Obesità/Compenso (BMI vs MSI)",
            style: TextStyle(
              color: Colors.amberAccent,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
        ),
        ListView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: data.length,
          itemBuilder: (context, index) {
            final p = data[index];
            return Card(
              margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              color: Colors.blueGrey[900]?.withOpacity(0.5),
              shape: RoundedRectangleBorder(
                side: BorderSide(
                  color: Colors.amberAccent.withOpacity(0.3),
                  width: 1,
                ),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Row(
                  children: [
                    // Colonna BMI (Massa)
                    Expanded(
                      child: Column(
                        children: [
                          const Text(
                            "MASSA (BMI)",
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 10,
                            ),
                          ),
                          Text(
                            "${p['Derived_BMI'].toStringAsFixed(1)}",
                            style: const TextStyle(
                              color: Colors.orangeAccent,
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                    // Icona di Mismatch (Centro)
                    const Padding(
                      padding: EdgeInsets.symmetric(horizontal: 8.0),
                      child: Icon(
                        Icons.sync_problem,
                        color: Colors.amberAccent,
                        size: 28,
                      ),
                    ),
                    // Colonna MSI (Risposta cardiaca)
                    Expanded(
                      child: Column(
                        children: [
                          const Text(
                            "RISPOSTA (MSI)",
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 10,
                            ),
                          ),
                          Text(
                            "${p['ModifiedShockIndex'].toStringAsFixed(2)}",
                            style: const TextStyle(
                              color: Colors.greenAccent,
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                    // Divider verticale e info rapide
                    Container(width: 1, height: 40, color: Colors.white10),
                    Padding(
                      padding: const EdgeInsets.only(left: 12.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            "ID: ${p['Patient ID']}",
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                            ),
                          ),
                          Text(
                            "HR: ${p['Heart Rate']} bpm",
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 11,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        ),
      ],
    );
  }
}

class OccultShockAlert extends StatelessWidget {
  final List<dynamic> data;

  const OccultShockAlert({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
          child: Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Colors.redAccent),
              SizedBox(width: 8),
              Text(
                "CRITICAL: OCCULT SHOCK (AGE < 40)",
                style: TextStyle(
                  color: Colors.redAccent,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.2,
                ),
              ),
            ],
          ),
        ),
        ListView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: data.length,
          itemBuilder: (context, index) {
            final p = data[index];
            return Container(
              margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.red.withOpacity(0.05),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: Colors.redAccent.withOpacity(0.5),
                  width: 1.5,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.redAccent.withOpacity(0.1),
                    blurRadius: 10,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: ListTile(
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                leading: CircleAvatar(
                  backgroundColor: Colors.redAccent,
                  child: const Icon(Icons.priority_high, color: Colors.white),
                ),
                title: Text(
                  "Paziente ID: ${p['Patient ID']}",
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                subtitle: Text(
                  "Età: ${p['Age']} anni | Temp: ${p['Body Temperature'].toStringAsFixed(1)}°C",
                  style: const TextStyle(color: Colors.white70),
                ),
                trailing: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    const Text(
                      "SHOCK INDEX",
                      style: TextStyle(
                        color: Colors.redAccent,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      "${p['ShockIndex'].toStringAsFixed(2)}",
                      style: const TextStyle(color: Colors.white, fontSize: 22),
                    ),
                  ],
                ),
              ),
            );
          },
        ),
      ],
    );
  }
}
