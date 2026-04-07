import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'screens/camera_screen.dart';
import 'services/yolo_inference.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Kick off camera + model load in parallel.
  final cameraFuture = availableCameras();
  final inference = YoloInference();
  final modelFuture = inference.loadModel();

  final cameras = await cameraFuture;
  await modelFuture;

  runApp(CacaoApp(cameras: cameras, inference: inference));
}

class CacaoApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  final YoloInference inference;

  const CacaoApp({
    super.key,
    required this.cameras,
    required this.inference,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cacao',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6B3E1A)),
        useMaterial3: true,
      ),
      home: CameraScreen(cameras: cameras, inference: inference),
    );
  }
}
