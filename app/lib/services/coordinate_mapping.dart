/// Coordinate mapping between three spaces:
///
///   1. model space       — 640×640 letterboxed input fed to the TFLite model
///   2. source photo      — the original camera JPEG, in sensor orientation
///   3. display canvas    — the Flutter widget area where we paint boxes
///
/// This file is the ONE place xy inversion can bite us. Keep it pure, keep it
/// small, cover it with tests (see test/coordinate_mapping_test.dart). If a
/// mapping feels clever, it's wrong.
library;

import 'dart:math' as math;
import 'dart:ui' show Size;

import '../models/detection.dart';

/// Result of letterboxing a (srcW × srcH) image into a (dstSize × dstSize) square
/// while preserving aspect ratio. The image is scaled by [scale] and pasted at
/// offset ([padX], [padY]) on a padded background.
class LetterboxParams {
  final int srcW;
  final int srcH;
  final int dstSize;
  final double scale;
  final double padX;
  final double padY;

  const LetterboxParams({
    required this.srcW,
    required this.srcH,
    required this.dstSize,
    required this.scale,
    required this.padX,
    required this.padY,
  });

  @override
  String toString() =>
      'Letterbox(src=${srcW}x$srcH, dst=$dstSize, scale=$scale, pad=($padX,$padY))';
}

/// Compute letterbox parameters for fitting (srcW × srcH) into (dstSize × dstSize).
LetterboxParams computeLetterbox(int srcW, int srcH, int dstSize) {
  assert(srcW > 0 && srcH > 0 && dstSize > 0);
  final double scale =
      math.min(dstSize / srcW, dstSize / srcH);
  final double newW = srcW * scale;
  final double newH = srcH * scale;
  final double padX = (dstSize - newW) / 2.0;
  final double padY = (dstSize - newH) / 2.0;
  return LetterboxParams(
    srcW: srcW,
    srcH: srcH,
    dstSize: dstSize,
    scale: scale,
    padX: padX,
    padY: padY,
  );
}

/// Map a bbox expressed in the 640×640 model space back to the source photo
/// coordinate system. Inverse of the letterbox transform.
BBox modelToSource(BBox modelBox, LetterboxParams lb) {
  final double x1 = (modelBox.x1 - lb.padX) / lb.scale;
  final double y1 = (modelBox.y1 - lb.padY) / lb.scale;
  final double x2 = (modelBox.x2 - lb.padX) / lb.scale;
  final double y2 = (modelBox.y2 - lb.padY) / lb.scale;
  return BBox(x1, y1, x2, y2).clamp(lb.srcW.toDouble(), lb.srcH.toDouble());
}

/// Inverse direction, useful for tests. Maps a source-photo bbox into the
/// 640×640 letterboxed model space.
BBox sourceToModel(BBox sourceBox, LetterboxParams lb) {
  final double x1 = sourceBox.x1 * lb.scale + lb.padX;
  final double y1 = sourceBox.y1 * lb.scale + lb.padY;
  final double x2 = sourceBox.x2 * lb.scale + lb.padX;
  final double y2 = sourceBox.y2 * lb.scale + lb.padY;
  return BBox(x1, y1, x2, y2);
}

/// Rotate a bbox that was defined in an unrotated source frame of size
/// (srcW × srcH) by [rotationDegrees] clockwise (0/90/180/270 only).
///
/// After rotation, the bbox is expressed in the rotated image frame, whose
/// dimensions are:
///   - 0°  / 180°: srcW × srcH
///   - 90° / 270°: srcH × srcW
BBox rotateBoxClockwise(BBox b, int srcW, int srcH, int rotationDegrees) {
  final int rot = ((rotationDegrees % 360) + 360) % 360;
  switch (rot) {
    case 0:
      return b;
    case 90:
      // (x, y) -> (srcH - y, x); swap (x1,x2) order after flip
      return BBox(
        srcH - b.y2, b.x1,
        srcH - b.y1, b.x2,
      );
    case 180:
      return BBox(
        srcW - b.x2, srcH - b.y2,
        srcW - b.x1, srcH - b.y1,
      );
    case 270:
      // (x, y) -> (y, srcW - x)
      return BBox(
        b.y1, srcW - b.x2,
        b.y2, srcW - b.x1,
      );
    default:
      throw ArgumentError('rotationDegrees must be a multiple of 90, got $rotationDegrees');
  }
}

/// Map a source-photo bbox to the display canvas coordinate system, assuming
/// the photo is rendered with BoxFit.contain inside [canvasSize] (i.e. the
/// Flutter `Image.file(...)` default when wrapped in a `FittedBox` or
/// `AspectRatio`). Returns the bbox in canvas-space pixels.
///
/// If [rotationDegrees] is non-zero, the source frame is rotated clockwise
/// first (matching how the image is rendered on screen).
BBox sourceToCanvas(
  BBox sourceBox,
  Size photoSize,
  Size canvasSize, {
  int rotationDegrees = 0,
}) {
  // 1. Apply rotation to both the bbox and the photo frame size.
  final int rot = ((rotationDegrees % 360) + 360) % 360;
  final BBox rotated = rotateBoxClockwise(
    sourceBox,
    photoSize.width.toInt(),
    photoSize.height.toInt(),
    rot,
  );
  final double rotW = (rot == 90 || rot == 270)
      ? photoSize.height
      : photoSize.width;
  final double rotH = (rot == 90 || rot == 270)
      ? photoSize.width
      : photoSize.height;

  // 2. BoxFit.contain: scale to fit, letterbox with equal padding on one axis.
  final double scale = math.min(canvasSize.width / rotW, canvasSize.height / rotH);
  final double drawnW = rotW * scale;
  final double drawnH = rotH * scale;
  final double padX = (canvasSize.width - drawnW) / 2.0;
  final double padY = (canvasSize.height - drawnH) / 2.0;

  return BBox(
    rotated.x1 * scale + padX,
    rotated.y1 * scale + padY,
    rotated.x2 * scale + padX,
    rotated.y2 * scale + padY,
  );
}
