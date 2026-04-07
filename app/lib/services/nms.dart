/// Non-max suppression, pure Dart. Operates on Detection lists in any coordinate
/// space (model or source — caller decides).
library;

import '../models/detection.dart';

/// Intersection-over-union of two boxes. Returns 0 if disjoint or zero-area.
double iou(BBox a, BBox b) {
  final double interX1 = a.x1 > b.x1 ? a.x1 : b.x1;
  final double interY1 = a.y1 > b.y1 ? a.y1 : b.y1;
  final double interX2 = a.x2 < b.x2 ? a.x2 : b.x2;
  final double interY2 = a.y2 < b.y2 ? a.y2 : b.y2;

  final double iw = interX2 - interX1;
  final double ih = interY2 - interY1;
  if (iw <= 0 || ih <= 0) return 0.0;

  final double inter = iw * ih;
  final double union = a.area + b.area - inter;
  if (union <= 0) return 0.0;
  return inter / union;
}

/// Greedy NMS. Input is mutated-safe (sorted copy internally). Same behaviour
/// as ultralytics default: per-class suppression then merge.
List<Detection> nms(
  List<Detection> input, {
  double iouThreshold = 0.45,
}) {
  if (input.isEmpty) return const [];

  // Group by class so we only suppress within-class.
  final Map<int, List<Detection>> byClass = {};
  for (final d in input) {
    byClass.putIfAbsent(d.classId, () => []).add(d);
  }

  final List<Detection> kept = [];
  for (final entry in byClass.entries) {
    final dets = [...entry.value]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    final List<bool> suppressed = List<bool>.filled(dets.length, false);
    for (int i = 0; i < dets.length; i++) {
      if (suppressed[i]) continue;
      kept.add(dets[i]);
      for (int j = i + 1; j < dets.length; j++) {
        if (suppressed[j]) continue;
        if (iou(dets[i].bbox, dets[j].bbox) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
  }

  // Final sort by confidence desc for deterministic display ordering.
  kept.sort((a, b) => b.confidence.compareTo(a.confidence));
  return kept;
}
