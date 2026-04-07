/// YOLOv8n TFLite inference pipeline.
///
/// Loads the FP16 model from assets, runs with the XNNPACK CPU delegate, and
/// decodes the raw output tensor into a list of [Detection] in source-photo
/// coordinates.
library;

import 'dart:developer' as dev;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection.dart';
import 'coordinate_mapping.dart';
import 'nms.dart';

const String _kModelAsset = 'assets/models/best_float16.tflite';
const String _kLabelsAsset = 'assets/labels.txt';
const int _kInputSize = 640;
const double _kConfThreshold = 0.25;
const double _kIouThreshold = 0.45;

/// Decoded YOLOv8 raw output, in 640×640 model space, before NMS/remap.
/// Public (no underscore) so the top-level [decodeYoloOutput] can be unit tested.
class RawDetection {
  final BBox box;
  final int classId;
  final double confidence;
  const RawDetection(this.box, this.classId, this.confidence);
}

class YoloInference {
  Interpreter? _interpreter;
  List<String> _labels = const [];

  // Cached tensor shape info (discovered at load time to handle NHWC vs NCHW
  // and output layout variants).
  List<int> _inputShape = const [];
  List<int> _outputShape = const [];
  bool _outputChannelsFirst = true; // (1, 8, 8400) = true, (1, 8400, 8) = false

  int get inputSize => _kInputSize;
  List<String> get labels => _labels;
  bool get isLoaded => _interpreter != null;

  Future<void> loadModel() async {
    // Labels
    final String rawLabels = await rootBundle.loadString(_kLabelsAsset);
    _labels = rawLabels
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList(growable: false);

    // Interpreter with XNNPACK delegate for optimized CPU inference on ARM.
    final options = InterpreterOptions()..addDelegate(XNNPackDelegate());
    _interpreter = await Interpreter.fromAsset(_kModelAsset, options: options);

    _inputShape = _interpreter!.getInputTensor(0).shape;
    _outputShape = _interpreter!.getOutputTensor(0).shape;
    dev.log('yolo: input=$_inputShape output=$_outputShape', name: 'yolo');

    // Detect output layout: (1, 4+nc, anchors) vs (1, anchors, 4+nc).
    // For v1 with 4 classes, 4+nc = 8.
    if (_outputShape.length == 3) {
      // [1, A, B]: if A == 8, channels-first; if B == 8, channels-last.
      if (_outputShape[1] == 4 + _labels.length) {
        _outputChannelsFirst = true;
      } else if (_outputShape[2] == 4 + _labels.length) {
        _outputChannelsFirst = false;
      } else {
        throw StateError(
          'Unexpected YOLO output shape $_outputShape for ${_labels.length} classes',
        );
      }
    } else {
      throw StateError('YOLO output tensor must be rank 3, got $_outputShape');
    }
  }

  /// Full inference pipeline: preprocess → run → decode → NMS → remap to
  /// source-photo coordinates.
  ///
  /// [photo] must already be rotated to the orientation the user expects to
  /// see on screen (call [img.bakeOrientation] on the decoded JPEG first).
  Future<List<Detection>> infer(img.Image photo) async {
    final itp = _interpreter;
    if (itp == null) {
      throw StateError('YoloInference.loadModel() not called yet');
    }

    final LetterboxParams lb =
        computeLetterbox(photo.width, photo.height, _kInputSize);

    // 1. Letterbox into 640×640 with gray padding (114,114,114 — YOLO default).
    final img.Image canvas = img.Image(
      width: _kInputSize,
      height: _kInputSize,
      numChannels: 3,
    );
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
    final img.Image resized = img.copyResize(
      photo,
      width: (photo.width * lb.scale).round(),
      height: (photo.height * lb.scale).round(),
      interpolation: img.Interpolation.linear,
    );
    img.compositeImage(
      canvas,
      resized,
      dstX: lb.padX.round(),
      dstY: lb.padY.round(),
    );

    // 2. Normalize to float32 NHWC [1, 640, 640, 3] in [0,1].
    final Float32List flat =
        Float32List(1 * _kInputSize * _kInputSize * 3);
    int p = 0;
    for (int y = 0; y < _kInputSize; y++) {
      for (int x = 0; x < _kInputSize; x++) {
        final px = canvas.getPixel(x, y);
        flat[p++] = px.r / 255.0;
        flat[p++] = px.g / 255.0;
        flat[p++] = px.b / 255.0;
      }
    }
    final input = flat.reshape([1, _kInputSize, _kInputSize, 3]);

    // 3. Allocate output with the shape reported by the interpreter.
    final output = List.filled(
      _outputShape.reduce((a, b) => a * b),
      0.0,
    ).reshape(_outputShape);

    // 4. Run.
    itp.run(input, output);

    // 5. Decode raw → NMS → source coords.
    final raw = decodeYoloOutput(
      output,
      _outputShape,
      _outputChannelsFirst,
      _labels.length,
      _kConfThreshold,
    );
    final List<Detection> named = raw
        .map((r) => Detection(
              classId: r.classId,
              label: _labels[r.classId],
              confidence: r.confidence,
              bbox: r.box,
            ))
        .toList();
    final suppressed = nms(named, iouThreshold: _kIouThreshold);

    // 6. Remap 640-space → source photo coords.
    return suppressed
        .map((d) => d.copyWith(bbox: modelToSource(d.bbox, lb)))
        .toList();
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}

/// Pure decoder for the YOLOv8 head output. Factored out so it can be unit
/// tested with a fake tensor and no interpreter.
///
/// [output] is the nested list returned by tflite_flutter (3-d). [shape] is
/// [1, 4+nc, anchors] if [channelsFirst] is true, else [1, anchors, 4+nc].
List<RawDetection> decodeYoloOutput(
  dynamic output,
  List<int> shape,
  bool channelsFirst,
  int numClasses,
  double confThreshold,
) {
  final int numAnchors = channelsFirst ? shape[2] : shape[1];
  final List<RawDetection> out = [];

  // Helper: reads (channelIdx, anchorIdx) regardless of layout.
  double read(int c, int a) {
    if (channelsFirst) {
      return (output[0][c][a] as num).toDouble();
    } else {
      return (output[0][a][c] as num).toDouble();
    }
  }

  for (int a = 0; a < numAnchors; a++) {
    // 4 bbox + nc class scores.
    int bestCls = -1;
    double bestScore = -1;
    for (int c = 0; c < numClasses; c++) {
      final double s = read(4 + c, a);
      if (s > bestScore) {
        bestScore = s;
        bestCls = c;
      }
    }
    if (bestScore < confThreshold) continue;

    final double cx = read(0, a);
    final double cy = read(1, a);
    final double w = read(2, a);
    final double h = read(3, a);
    final BBox box = BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2);
    out.add(RawDetection(box, bestCls, bestScore));
  }
  return out;
}
