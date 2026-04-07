/// CustomPainter that draws detection bounding boxes on top of a photo
/// rendered with BoxFit.contain inside the painter's canvas.
///
/// All detections received are expected to be in SOURCE PHOTO coordinates.
/// The painter handles the photo-to-canvas mapping (letterbox BoxFit.contain +
/// rotation) via [sourceToCanvas].
library;

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../services/coordinate_mapping.dart';

class BBoxOverlayPainter extends CustomPainter {
  final List<Detection> detections;
  final Size photoSize;
  final int rotationDegrees;

  BBoxOverlayPainter({
    required this.detections,
    required this.photoSize,
    this.rotationDegrees = 0,
  });

  static const Map<String, Color> _classColors = {
    'healthy': Color(0xFF2ECC71),
    'cssvd': Color(0xFFE74C3C),
    'anthracnose': Color(0xFFF39C12),
    'black_pod': Color(0xFF8E44AD),
  };

  @override
  void paint(Canvas canvas, Size size) {
    for (final d in detections) {
      final BBox mapped = sourceToCanvas(
        d.bbox,
        photoSize,
        size,
        rotationDegrees: rotationDegrees,
      );
      final color = _classColors[d.label] ?? Colors.red;

      final rect = Rect.fromLTRB(mapped.x1, mapped.y1, mapped.x2, mapped.y2);
      final stroke = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;
      canvas.drawRect(rect, stroke);

      // Label background + text.
      final text = '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%';
      final tp = TextPainter(
        text: TextSpan(
          text: text,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelBg = Rect.fromLTWH(
        mapped.x1,
        (mapped.y1 - tp.height - 4).clamp(0.0, size.height),
        tp.width + 8,
        tp.height + 4,
      );
      canvas.drawRect(labelBg, Paint()..color = color);
      tp.paint(canvas, Offset(labelBg.left + 4, labelBg.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant BBoxOverlayPainter old) =>
      old.detections != detections ||
      old.photoSize != photoSize ||
      old.rotationDegrees != rotationDegrees;
}
