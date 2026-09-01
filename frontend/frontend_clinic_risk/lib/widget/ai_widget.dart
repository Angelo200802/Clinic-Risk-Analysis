import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

class AiExplanationPanel extends StatefulWidget {
  final int? patientId;
  final String streamingText;
  final String? completedText;
  final bool isStreaming;

  const AiExplanationPanel({
    super.key,
    required this.patientId,
    required this.streamingText,
    required this.completedText,
    required this.isStreaming,
  });

  @override
  State<AiExplanationPanel> createState() => _AiExplanationPanelState();
}

class _AiExplanationPanelState extends State<AiExplanationPanel> {
  final ScrollController _scrollCtrl = ScrollController();

  @override
  void didUpdateWidget(AiExplanationPanel old) {
    super.didUpdateWidget(old);
    if (widget.streamingText != old.streamingText) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scrollCtrl.hasClients) {
          _scrollCtrl.animateTo(
            _scrollCtrl.position.maxScrollExtent,
            duration: const Duration(milliseconds: 20),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  @override
  void dispose() {
    _scrollCtrl.dispose();
    super.dispose();
  }

  String get _displayText => widget.isStreaming
      ? widget.streamingText
      : (widget.completedText ?? widget.streamingText);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Row(
              children: [
                Icon(Icons.auto_awesome, color: Color(0xFFAFA9EC), size: 14),
                SizedBox(width: 8),
                Text(
                  "AI CLINICAL SUMMARY",
                  style: TextStyle(
                    color: Colors.white54,
                    fontSize: 10,
                    letterSpacing: 1.2,
                  ),
                ),
              ],
            ),
            if (widget.isStreaming)
              const _DotIndicator()
            else if (widget.completedText != null)
              const Icon(
                Icons.check_circle_outline,
                size: 14,
                color: Colors.greenAccent,
              ),
          ],
        ),
        const Divider(color: Colors.white10, height: 20),

        // Contenuto
        Expanded(
          child: widget.patientId == null
              ? _buildPlaceholder()
              : _buildContent(),
        ),
      ],
    );
  }

  Widget _buildPlaceholder() {
    return const Center(
      child: Text(
        "Seleziona paziente",
        style: TextStyle(
          color: Colors.white10,
          fontSize: 12,
          fontStyle: FontStyle.italic,
        ),
      ),
    );
  }

  Widget _buildContent() {
    if (_displayText.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.auto_awesome,
              size: 32,
              color: Colors.white.withOpacity(0.05),
            ),
            const SizedBox(height: 12),
            Text(
              "Premi il bottone per generare\nun'analisi AI del paziente",
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withOpacity(0.15),
                fontSize: 12,
              ),
            ),
          ],
        ),
      );
    }

    return Stack(
      children: [
        Positioned.fill(
          child: SingleChildScrollView(
            controller: _scrollCtrl,
            physics: const BouncingScrollPhysics(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                MarkdownBody(
                  data: _displayText,
                  styleSheet: MarkdownStyleSheet(
                    p: const TextStyle(
                      color: Colors.white70,
                      fontSize: 13,
                      height: 1.7,
                    ),
                    h1: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                      height: 1.5,
                    ),
                    h2: const TextStyle(
                      color: Colors.white,
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      height: 1.5,
                    ),
                    h3: const TextStyle(
                      color: Color(0xFFAFA9EC),
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      height: 1.5,
                    ),
                    strong: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                    ),
                    em: const TextStyle(
                      color: Colors.white60,
                      fontStyle: FontStyle.italic,
                    ),
                    code: const TextStyle(
                      color: Color(0xFFAFA9EC),
                      fontSize: 12,
                      fontFamily: 'monospace',
                      backgroundColor: Color(0xFF2A2A2A),
                    ),
                    codeblockDecoration: BoxDecoration(
                      color: const Color(0xFF2A2A2A),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.white10),
                    ),
                    blockquote: const TextStyle(
                      color: Colors.white54,
                      fontSize: 13,
                      fontStyle: FontStyle.italic,
                    ),
                    blockquoteDecoration: const BoxDecoration(
                      border: Border(
                        left: BorderSide(color: Color(0xFF7F77DD), width: 3),
                      ),
                    ),
                    listBullet: const TextStyle(
                      color: Color(0xFFAFA9EC),
                      fontSize: 13,
                    ),
                    horizontalRuleDecoration: const BoxDecoration(
                      border: Border(
                        top: BorderSide(color: Colors.white10, width: 1),
                      ),
                    ),
                    tableHead: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 13,
                    ),
                    tableBody: const TextStyle(
                      color: Colors.white70,
                      fontSize: 13,
                    ),
                    tableBorder: TableBorder.all(
                      color: Colors.white10,
                      width: 0.5,
                    ),
                  ),
                ),
                if (widget.isStreaming) ...[
                  const SizedBox(height: 4),
                  const _BlinkingCursor(),
                ],
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _BlinkingCursor extends StatefulWidget {
  const _BlinkingCursor();

  @override
  State<_BlinkingCursor> createState() => _BlinkingCursorState();
}

class _BlinkingCursorState extends State<_BlinkingCursor>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => FadeTransition(
    opacity: _ctrl,
    child: Container(
      width: 2,
      height: 16,
      margin: const EdgeInsets.only(left: 2, bottom: 2),
      color: const Color(0xFFAFA9EC),
    ),
  );
}

class _DotIndicator extends StatefulWidget {
  const _DotIndicator();

  @override
  State<_DotIndicator> createState() => _DotIndicatorState();
}

class _DotIndicatorState extends State<_DotIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
    animation: _ctrl,
    builder: (_, __) => Row(
      children: List.generate(3, (i) {
        final opacity = ((_ctrl.value * 3) - i).clamp(0.0, 1.0);
        return Container(
          width: 5,
          height: 5,
          margin: const EdgeInsets.only(right: 4),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: const Color(0xFF7F77DD).withOpacity(0.3 + opacity * 0.7),
          ),
        );
      }),
    ),
  );
}
