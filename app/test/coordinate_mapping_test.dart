import 'dart:ui';

import 'package:cacao/models/detection.dart';
import 'package:cacao/services/coordinate_mapping.dart';
import 'package:flutter_test/flutter_test.dart';

void _bboxClose(BBox a, BBox b, {double tol = 1e-6}) {
  expect(a.x1, closeTo(b.x1, tol));
  expect(a.y1, closeTo(b.y1, tol));
  expect(a.x2, closeTo(b.x2, tol));
  expect(a.y2, closeTo(b.y2, tol));
}

void main() {
  group('computeLetterbox', () {
    test('square source → no padding', () {
      final lb = computeLetterbox(640, 640, 640);
      expect(lb.scale, 1.0);
      expect(lb.padX, 0);
      expect(lb.padY, 0);
    });

    test('landscape 1920x1080 → pad top/bottom', () {
      final lb = computeLetterbox(1920, 1080, 640);
      // scale = 640/1920 = 0.3333
      expect(lb.scale, closeTo(640 / 1920, 1e-9));
      // drawn height = 1080 * scale = 360; pad = (640-360)/2 = 140
      expect(lb.padX, 0);
      expect(lb.padY, closeTo(140, 1e-9));
    });

    test('portrait 1080x1920 → pad left/right', () {
      final lb = computeLetterbox(1080, 1920, 640);
      expect(lb.scale, closeTo(640 / 1920, 1e-9));
      expect(lb.padY, 0);
      expect(lb.padX, closeTo(140, 1e-9));
    });
  });

  group('letterbox round-trip (sourceToModel ∘ modelToSource = id)', () {
    for (final pair in [
      [1920, 1080],
      [1080, 1920],
      [1280, 720],
      [640, 640],
      [4000, 3000],
    ]) {
      test('${pair[0]}x${pair[1]}', () {
        final lb = computeLetterbox(pair[0], pair[1], 640);
        final sources = <BBox>[
          BBox(0, 0, pair[0].toDouble(), pair[1].toDouble()),
          const BBox(10, 20, 100, 200),
          BBox(pair[0] / 2, pair[1] / 2, pair[0] - 1, pair[1] - 1),
        ];
        for (final src in sources) {
          final round = modelToSource(sourceToModel(src, lb), lb);
          _bboxClose(round, src, tol: 1e-3);
        }
      });
    }
  });

  group('rotateBoxClockwise', () {
    // A box in the top-left corner of a 100×200 (wxh) frame.
    const src = BBox(10, 20, 30, 40);
    const W = 100;
    const H = 200;

    test('0° → unchanged', () {
      _bboxClose(rotateBoxClockwise(src, W, H, 0), src);
    });

    test('90° → right side of a 200×100 frame', () {
      // After 90° CW: rotated frame is H×W = 200×100.
      // Original top-left (10,20)..(30,40) → new (H-y2, x1)..(H-y1, x2)
      //                                    = (160, 10)..(180, 30)
      _bboxClose(
        rotateBoxClockwise(src, W, H, 90),
        const BBox(160, 10, 180, 30),
      );
    });

    test('180° → bottom-right of 100×200', () {
      // (W-x2, H-y2)..(W-x1, H-y1) = (70, 160)..(90, 180)
      _bboxClose(
        rotateBoxClockwise(src, W, H, 180),
        const BBox(70, 160, 90, 180),
      );
    });

    test('270° → bottom of a 200×100 frame', () {
      // (y1, W-x2)..(y2, W-x1) = (20, 70)..(40, 90)
      _bboxClose(
        rotateBoxClockwise(src, W, H, 270),
        const BBox(20, 70, 40, 90),
      );
    });

    test('360° normalizes to 0°', () {
      _bboxClose(rotateBoxClockwise(src, W, H, 360), src);
    });

    test('negative angle -90° = 270°', () {
      _bboxClose(
        rotateBoxClockwise(src, W, H, -90),
        rotateBoxClockwise(src, W, H, 270),
      );
    });
  });

  group('sourceToCanvas (BoxFit.contain)', () {
    test('full-photo bbox → full drawn area', () {
      const photo = Size(1920, 1080);
      const canvas = Size(800, 600);
      // photo aspect 1.78 > canvas aspect 1.33 → scaled to width 800, height 450
      // padY = (600-450)/2 = 75
      final mapped = sourceToCanvas(
        const BBox(0, 0, 1920, 1080),
        photo,
        canvas,
      );
      _bboxClose(mapped, const BBox(0, 75, 800, 525), tol: 1e-6);
    });

    test('center-point stays centered', () {
      const photo = Size(1000, 1000);
      const canvas = Size(500, 500);
      final mapped = sourceToCanvas(
        const BBox(450, 450, 550, 550),
        photo,
        canvas,
      );
      _bboxClose(mapped, const BBox(225, 225, 275, 275), tol: 1e-6);
    });

    test('top-left corner box stays top-left after rotation=0', () {
      const photo = Size(1000, 500);
      const canvas = Size(1000, 1000);
      // photo scaled to 1000x500, padY = 250
      final mapped = sourceToCanvas(
        const BBox(0, 0, 100, 50),
        photo,
        canvas,
      );
      _bboxClose(mapped, const BBox(0, 250, 100, 300), tol: 1e-6);
    });

    test('rotation 90°: top-left source → top-right of canvas after rotation', () {
      const photo = Size(1000, 500);
      const canvas = Size(500, 1000);
      // After 90° CW, rotated frame is 500×1000 (w×h). A top-left source bbox
      // (0,0)-(100,50) becomes, in the rotated frame, (H-50, 0)-(H-0, 100)
      // = (450, 0)-(500, 100). That frame maps 1:1 onto the 500×1000 canvas
      // (same aspect), so the canvas bbox should be identical.
      final mapped = sourceToCanvas(
        const BBox(0, 0, 100, 50),
        photo,
        canvas,
        rotationDegrees: 90,
      );
      _bboxClose(mapped, const BBox(450, 0, 500, 100), tol: 1e-6);
    });

    test('portrait photo vs landscape photo do not silently swap w/h', () {
      // A box at (0,0)-(100,100) in a 1080×1920 photo rendered on a 500×500
      // canvas. 1080×1920 aspect 0.5625 < 1 → scale to height 500, width ≈ 281
      // padX ≈ 109.375. Drawn box = (0*scale+padX, 0*scale+padY)..(100*scale+padX, 100*scale+padY)
      //   scale = 500/1920 ≈ 0.2604
      //   padX = (500 - 1080*scale)/2 = (500 - 281.25)/2 ≈ 109.375
      //   padY = 0
      const photo = Size(1080, 1920);
      const canvas = Size(500, 500);
      final mapped = sourceToCanvas(const BBox(0, 0, 100, 100), photo, canvas);
      final scale = 500 / 1920;
      final padX = (500 - 1080 * scale) / 2;
      _bboxClose(
        mapped,
        BBox(padX, 0, 100 * scale + padX, 100 * scale),
        tol: 1e-6,
      );
    });
  });
}
