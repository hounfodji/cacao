import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../services/yolo_inference.dart';
import '../models/detection.dart';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  final YoloInference inference;

  const CameraScreen({
    super.key,
    required this.cameras,
    required this.inference,
  });

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  bool _initializing = true;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    if (widget.cameras.isEmpty) {
      setState(() => _initializing = false);
      return;
    }
    final back = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => widget.cameras.first,
    );
    final controller = CameraController(
      back,
      ResolutionPreset.high, // ~1280×720, suffisant pour le modèle 640
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    await controller.initialize();
    if (!mounted) return;
    setState(() {
      _controller = controller;
      _initializing = false;
    });
  }

  Future<void> _capture() async {
    final c = _controller;
    if (c == null || !c.value.isInitialized || _busy) return;
    setState(() => _busy = true);

    try {
      final XFile file = await c.takePicture();
      final bytes = await File(file.path).readAsBytes();
      img.Image? decoded = img.decodeImage(bytes);
      if (decoded == null) {
        throw const FormatException('Impossible de décoder la photo');
      }
      // Apply EXIF orientation now so downstream code sees an upright image
      // and we can pass rotationDegrees=0 to the painter.
      decoded = img.bakeOrientation(decoded);

      final List<Detection> dets = await widget.inference.infer(decoded);

      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            photoPath: file.path,
            photoWidth: decoded!.width,
            photoHeight: decoded.height,
            detections: dets,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Erreur: $e')),
      );
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_initializing) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }
    final c = _controller;
    if (c == null || !c.value.isInitialized) {
      return const Scaffold(
        body: Center(
          child: Padding(
            padding: EdgeInsets.all(24),
            child: Text(
              'Aucune caméra disponible. Vérifiez les permissions.',
              textAlign: TextAlign.center,
            ),
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Diagnostic cacao')),
      body: Column(
        children: [
          Expanded(child: CameraPreview(c)),
          Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                const Text(
                  'Cadrez la cabosse dans la photo puis appuyez pour diagnostiquer',
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: _busy ? null : _capture,
                    icon: _busy
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.camera_alt),
                    label: Text(_busy ? 'Analyse...' : 'Diagnostiquer'),
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
