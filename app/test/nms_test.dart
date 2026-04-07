import 'package:cacao/models/detection.dart';
import 'package:cacao/services/nms.dart';
import 'package:flutter_test/flutter_test.dart';

Detection _d(double conf, double x1, double y1, double x2, double y2,
    {int classId = 0, String label = 'healthy'}) {
  return Detection(
    classId: classId,
    label: label,
    confidence: conf,
    bbox: BBox(x1, y1, x2, y2),
  );
}

void main() {
  group('iou', () {
    test('identical boxes → 1.0', () {
      expect(iou(const BBox(0, 0, 10, 10), const BBox(0, 0, 10, 10)),
          closeTo(1.0, 1e-9));
    });

    test('disjoint boxes → 0', () {
      expect(iou(const BBox(0, 0, 10, 10), const BBox(20, 20, 30, 30)), 0.0);
    });

    test('touching boxes → 0', () {
      expect(iou(const BBox(0, 0, 10, 10), const BBox(10, 0, 20, 10)), 0.0);
    });

    test('half overlap → 1/3', () {
      // A = 100, B = 100, inter = 50, union = 150 → 1/3
      final v = iou(const BBox(0, 0, 10, 10), const BBox(5, 0, 15, 10));
      expect(v, closeTo(1.0 / 3.0, 1e-9));
    });

    test('zero-area box → 0', () {
      expect(iou(const BBox(0, 0, 0, 0), const BBox(0, 0, 10, 10)), 0.0);
    });
  });

  group('nms', () {
    test('empty → empty', () {
      expect(nms(const []), isEmpty);
    });

    test('single detection passes through', () {
      final r = nms([_d(0.9, 0, 0, 10, 10)]);
      expect(r, hasLength(1));
      expect(r.first.confidence, 0.9);
    });

    test('overlapping same-class → keeps highest conf', () {
      final r = nms([
        _d(0.9, 0, 0, 10, 10),
        _d(0.8, 1, 1, 11, 11), // IoU ~0.68 > 0.45
        _d(0.7, 2, 2, 12, 12),
      ]);
      expect(r, hasLength(1));
      expect(r.first.confidence, 0.9);
    });

    test('overlapping different classes → both kept (per-class NMS)', () {
      final r = nms([
        _d(0.9, 0, 0, 10, 10, classId: 0, label: 'healthy'),
        _d(0.8, 0, 0, 10, 10, classId: 1, label: 'cssvd'),
      ]);
      expect(r, hasLength(2));
    });

    test('non-overlapping same-class → both kept', () {
      final r = nms([
        _d(0.9, 0, 0, 10, 10),
        _d(0.8, 100, 100, 110, 110),
      ]);
      expect(r, hasLength(2));
    });

    test('output sorted by confidence desc', () {
      final r = nms([
        _d(0.5, 0, 0, 10, 10),
        _d(0.9, 100, 100, 110, 110),
        _d(0.7, 200, 200, 210, 210),
      ]);
      expect(r.map((d) => d.confidence).toList(), [0.9, 0.7, 0.5]);
    });
  });
}
