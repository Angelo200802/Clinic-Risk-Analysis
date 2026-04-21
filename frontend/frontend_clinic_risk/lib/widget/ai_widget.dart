// Crea widget/ai_explanation_panel.dart
import 'package:flutter/material.dart';

class AiExplanationPanel extends StatefulWidget {
  final int patientId;
  final String streamingText; // da _aiStreams[patientId]?.toString() ?? ''
  final String? completedText; // da _aiResponses[patientId]
  final bool isStreaming;
  final VoidCallback onClose;
  final VoidCallback onRegenerate;

  const AiExplanationPanel({
    super.key,
    required this.patientId,
    required this.streamingText,
    required this.completedText,
    required this.isStreaming,
    required this.onClose,
    required this.onRegenerate,
  });

  @override
  State<AiExplanationPanel> createState() => _AiExplanationPanelState();
}

class _AiExplanationPanelState extends State<AiExplanationPanel> {
  final ScrollController _scrollCtrl = ScrollController();

  @override
  void didUpdateWidget(AiExplanationPanel old) {
    super.didUpdateWidget(old);
    // Auto-scroll durante lo streaming
    if (widget.streamingText != old.streamingText) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scrollCtrl.hasClients) {
          _scrollCtrl.animateTo(
            _scrollCtrl.position.maxScrollExtent,
            duration: const Duration(milliseconds: 80),
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
    return Container(
      width: 360,
      decoration: const BoxDecoration(
        color: Color(0xFF1A1A1A),
        border: Border(left: BorderSide(color: Colors.white12)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 16, 12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Row(
                  children: [
                    Icon(
                      Icons.auto_awesome,
                      color: Color(0xFFAFA9EC),
                      size: 14,
                    ),
                    SizedBox(width: 8),
                    Text(
                      "AI CLINICAL SUMMARY",
                      style: TextStyle(
                        color: Color(0xFFAFA9EC),
                        fontSize: 10,
                        letterSpacing: 1.4,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
                IconButton(
                  icon: const Icon(
                    Icons.close,
                    size: 16,
                    color: Colors.white38,
                  ),
                  onPressed: widget.onClose,
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Text(
              "Patient #${widget.patientId}",
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          const Divider(color: Colors.white10, height: 24),

          // Testo in streaming
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.03),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.white10),
                ),
                child: SingleChildScrollView(
                  controller: _scrollCtrl,
                  physics: const BouncingScrollPhysics(),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Expanded(
                        child: Text(
                          _displayText,
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 13,
                            height: 1.7,
                          ),
                        ),
                      ),
                      if (widget.isStreaming) const _BlinkingCursor(),
                    ],
                  ),
                ),
              ),
            ),
          ),

          // Footer
          Padding(
            padding: const EdgeInsets.all(16),
            child: widget.isStreaming
                ? const Row(
                    children: [
                      _DotIndicator(),
                      SizedBox(width: 8),
                      Text(
                        "RECEIVING STREAM",
                        style: TextStyle(
                          color: Color(0xFF7F77DD),
                          fontSize: 10,
                          letterSpacing: 1.2,
                        ),
                      ),
                    ],
                  )
                : Row(
                    children: [
                      const Icon(
                        Icons.check_circle_outline,
                        size: 14,
                        color: Colors.greenAccent,
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        "Analysis complete",
                        style: TextStyle(
                          color: Colors.greenAccent,
                          fontSize: 11,
                        ),
                      ),
                      const Spacer(),
                      GestureDetector(
                        onTap: widget.onRegenerate,
                        child: const Text(
                          "Regenerate",
                          style: TextStyle(
                            color: Color(0xFFAFA9EC),
                            fontSize: 11,
                            decoration: TextDecoration.underline,
                          ),
                        ),
                      ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }
}

// Cursore lampeggiante
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

// Tre pallini animati
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
