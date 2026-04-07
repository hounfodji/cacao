/// Domain types for detection results.
///
/// Coordinates are always xyxy (not cxcywh). The space (model-640, source photo,
/// or canvas) must be tracked by the caller — see `coordinate_mapping.dart`.
library;

class BBox {
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  const BBox(this.x1, this.y1, this.x2, this.y2);

  double get width => x2 - x1;
  double get height => y2 - y1;
  double get area => width * height;
  double get cx => (x1 + x2) / 2;
  double get cy => (y1 + y2) / 2;

  /// Clamp to [0, w] × [0, h]. Useful after coordinate remaps to handle
  /// floating-point overflow past the image edges.
  BBox clamp(double w, double h) {
    return BBox(
      x1.clamp(0.0, w),
      y1.clamp(0.0, h),
      x2.clamp(0.0, w),
      y2.clamp(0.0, h),
    );
  }

  @override
  String toString() =>
      'BBox(${x1.toStringAsFixed(1)}, ${y1.toStringAsFixed(1)}, '
      '${x2.toStringAsFixed(1)}, ${y2.toStringAsFixed(1)})';

  @override
  bool operator ==(Object other) =>
      other is BBox &&
      x1 == other.x1 &&
      y1 == other.y1 &&
      x2 == other.x2 &&
      y2 == other.y2;

  @override
  int get hashCode => Object.hash(x1, y1, x2, y2);
}

class Detection {
  final int classId;
  final String label;
  final double confidence;
  final BBox bbox;

  const Detection({
    required this.classId,
    required this.label,
    required this.confidence,
    required this.bbox,
  });

  Detection copyWith({BBox? bbox}) => Detection(
        classId: classId,
        label: label,
        confidence: confidence,
        bbox: bbox ?? this.bbox,
      );

  @override
  String toString() =>
      '$label (${(confidence * 100).toStringAsFixed(1)}%) $bbox';
}
