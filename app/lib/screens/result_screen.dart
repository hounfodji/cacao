import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../models/detection.dart';
import '../widgets/bbox_overlay.dart';

class ResultScreen extends StatefulWidget {
  final String photoPath;
  final int photoWidth;
  final int photoHeight;
  final List<Detection> detections;

  const ResultScreen({
    super.key,
    required this.photoPath,
    required this.photoWidth,
    required this.photoHeight,
    required this.detections,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  Map<String, dynamic>? _advice;

  @override
  void initState() {
    super.initState();
    _loadAdvice();
  }

  Future<void> _loadAdvice() async {
    final raw = await rootBundle.loadString('assets/advice/advice_fr.json');
    setState(() => _advice = json.decode(raw) as Map<String, dynamic>);
  }

  String _title() {
    if (widget.detections.isEmpty) return 'Aucune maladie détectée';
    final top = widget.detections.first;
    final a = _advice?[top.label] as Map<String, dynamic>?;
    return a?['title'] as String? ?? top.label;
  }

  String _body() {
    if (widget.detections.isEmpty) {
      return "Aucune cabosse ou maladie n'a été reconnue avec une confiance suffisante. "
          "Rapprochez-vous et cadrez la cabosse en plein centre, avec une bonne lumière.";
    }
    final top = widget.detections.first;
    final a = _advice?[top.label] as Map<String, dynamic>?;
    return a?['body'] as String? ??
        'Résultat: ${top.label} (${(top.confidence * 100).toStringAsFixed(0)}%)';
  }

  @override
  Widget build(BuildContext context) {
    final photoSize =
        Size(widget.photoWidth.toDouble(), widget.photoHeight.toDouble());

    return Scaffold(
      appBar: AppBar(title: const Text('Résultat')),
      body: Column(
        children: [
          Expanded(
            flex: 3,
            child: LayoutBuilder(
              builder: (context, constraints) {
                final canvasSize =
                    Size(constraints.maxWidth, constraints.maxHeight);
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.file(
                      File(widget.photoPath),
                      fit: BoxFit.contain,
                    ),
                    // The overlay must use the SAME canvas size the image is
                    // laid out in, and the SAME fit (BoxFit.contain).
                    Positioned.fill(
                      child: CustomPaint(
                        painter: BBoxOverlayPainter(
                          detections: widget.detections,
                          photoSize: photoSize,
                          // We baked EXIF orientation upstream, so 0 here.
                          rotationDegrees: 0,
                        ),
                        size: canvasSize,
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
          Expanded(
            flex: 2,
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _title(),
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 8),
                  if (widget.detections.isNotEmpty)
                    Text(
                      'Confiance: ${(widget.detections.first.confidence * 100).toStringAsFixed(0)}%',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  const SizedBox(height: 16),
                  Text(_body()),
                  const SizedBox(height: 16),
                  if (widget.detections.length > 1) ...[
                    const Divider(),
                    Text(
                      'Autres détections',
                      style: Theme.of(context).textTheme.titleSmall,
                    ),
                    for (final d in widget.detections.skip(1))
                      Text('  • ${d.label} (${(d.confidence * 100).toStringAsFixed(0)}%)'),
                  ],
                ],
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                icon: const Icon(Icons.camera_alt),
                label: const Text('Nouvelle photo'),
                onPressed: () => Navigator.of(context).pop(),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
