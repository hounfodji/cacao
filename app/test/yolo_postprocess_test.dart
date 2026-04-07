import 'package:cacao/services/yolo_inference.dart';
import 'package:flutter_test/flutter_test.dart';

/// Build a fake YOLOv8 output tensor shape [1, 4+nc, anchors] filled with zeros,
/// then inject [detections] as (anchor_idx, cx, cy, w, h, classId, score).
List<List<List<double>>> makeFakeTensor({
  required int nc,
  required int anchors,
  required List<List<double>> detections,
}) {
  final channels = 4 + nc;
  final t = List.generate(
    1,
    (_) => List.generate(
      channels,
      (_) => List<double>.filled(anchors, 0.0),
    ),
  );
  for (final d in detections) {
    final int a = d[0].toInt();
    t[0][0][a] = d[1]; // cx
    t[0][1][a] = d[2]; // cy
    t[0][2][a] = d[3]; // w
    t[0][3][a] = d[4]; // h
    final int cls = d[5].toInt();
    final double score = d[6];
    t[0][4 + cls][a] = score;
  }
  return t;
}

void main() {
  group('decodeYoloOutput (channels-first)', () {
    test('zero tensor → no detections', () {
      final t = makeFakeTensor(nc: 4, anchors: 10, detections: []);
      final result = decodeYoloOutput(t, [1, 8, 10], true, 4, 0.25);
      expect(result, isEmpty);
    });

    test('one clear detection above threshold', () {
      final t = makeFakeTensor(nc: 4, anchors: 10, detections: [
        // anchor 3: (cx=100, cy=200, w=50, h=60) class=2 score=0.9
        [3, 100, 200, 50, 60, 2, 0.9],
      ]);
      final result = decodeYoloOutput(t, [1, 8, 10], true, 4, 0.25);
      expect(result, hasLength(1));
      expect(result.first.classId, 2);
      expect(result.first.confidence, closeTo(0.9, 1e-9));
      // xyxy = (75, 170, 125, 230)
      expect(result.first.box.x1, closeTo(75, 1e-9));
      expect(result.first.box.y1, closeTo(170, 1e-9));
      expect(result.first.box.x2, closeTo(125, 1e-9));
      expect(result.first.box.y2, closeTo(230, 1e-9));
    });

    test('detection below threshold is dropped', () {
      final t = makeFakeTensor(nc: 4, anchors: 5, detections: [
        [0, 100, 100, 20, 20, 0, 0.2], // below 0.25
      ]);
      final result = decodeYoloOutput(t, [1, 8, 5], true, 4, 0.25);
      expect(result, isEmpty);
    });

    test('argmax picks highest class score', () {
      final t = makeFakeTensor(nc: 4, anchors: 5, detections: []);
      // Manually inject mixed scores on anchor 2: class 1=0.3, class 3=0.8
      t[0][0][2] = 50;
      t[0][1][2] = 50;
      t[0][2][2] = 10;
      t[0][3][2] = 10;
      t[0][4 + 1][2] = 0.3;
      t[0][4 + 3][2] = 0.8;
      final result = decodeYoloOutput(t, [1, 8, 5], true, 4, 0.25);
      expect(result, hasLength(1));
      expect(result.first.classId, 3);
      expect(result.first.confidence, closeTo(0.8, 1e-9));
    });
  });

  group('decodeYoloOutput (channels-last)', () {
    test('same detection works in (1, anchors, 4+nc) layout', () {
      // Build [1, anchors=5, 8] tensor manually.
      const anchors = 5;
      const channels = 8;
      final t = List.generate(
        1,
        (_) => List.generate(
          anchors,
          (_) => List<double>.filled(channels, 0.0),
        ),
      );
      // anchor 2: (cx=100, cy=200, w=50, h=60), class=2, score=0.9
      t[0][2][0] = 100;
      t[0][2][1] = 200;
      t[0][2][2] = 50;
      t[0][2][3] = 60;
      t[0][2][4 + 2] = 0.9;

      final result = decodeYoloOutput(t, [1, anchors, channels], false, 4, 0.25);
      expect(result, hasLength(1));
      expect(result.first.classId, 2);
      expect(result.first.box.x1, closeTo(75, 1e-9));
    });
  });
}
