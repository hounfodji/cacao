# Cacao v1 — Flutter Android app

App Android offline de détection de maladies du cacao pour petit producteur ouest-africain.
Modèle: YOLOv8n TFLite FP16, 4 classes (`healthy`, `cssvd`, `anthracnose`, `black_pod`),
entraîné sur ~21k images (dédupliquées, stratifiées par source). mAP@0.5 global 0.803,
West Africa subset 0.751.

## Setup initial (une fois, sur ta machine avec Flutter installé)

Le répertoire `app/` contient déjà `pubspec.yaml`, `lib/`, `assets/`, `test/`. Il manque
juste le scaffold natif Android (généré par `flutter create`):

```bash
cd app
# 1. Générer le scaffold natif Android (garde nos fichiers existants intacts)
flutter create --platforms=android --org com.cacaoai --project-name cacao .

# 2. Copier le modèle TFLite entraîné dans les assets
cp ../runs/detect/ml/runs/v1_yolov8n/weights/best_saved_model/best_float16.tflite \
   assets/models/best_float16.tflite

# 3. Fixer minSdkVersion dans android/app/build.gradle :
#    minSdkVersion 24   (pour tflite_flutter + Android 7+)

# 4. Installer les deps et lancer les tests
flutter pub get
flutter test
```

Les tests doivent tous passer **avant** de déployer sur device. Ils couvrent
la partie la plus risquée: le mapping de coordonnées bbox (letterbox ↔ photo ↔ canvas)
et la NMS. Aucun test ne nécessite un device.

## Déploiement sur device

```bash
# téléphone Android branché en USB debugging
flutter run --release > /tmp/app_logs.txt 2>&1
```

`--release` est important: en debug l'inférence est 3-5× plus lente et fausse
les mesures de perf.

## Sanity checks E2E (manuel sur device)

Voir `../../.claude/plans/robust-sleeping-whistle.md` section "Vérification end-to-end"
pour la liste des 8 checks à faire (permission caméra, bbox visuellement juste,
rotation portrait/landscape, 10 photos d'affilée sans OOM, etc.).

Le critère de succès v1: 8/8 checks OK sur 1 device Android réel.

## Structure

```
app/
├── pubspec.yaml
├── assets/
│   ├── models/best_float16.tflite   (copié manuellement après flutter create)
│   ├── labels.txt                   (4 classes)
│   └── advice/advice_fr.json        (conseil par maladie, à réviser par agronome)
├── lib/
│   ├── main.dart
│   ├── models/detection.dart
│   ├── services/
│   │   ├── nms.dart                 (non-max suppression Dart pur)
│   │   ├── coordinate_mapping.dart  (letterbox + source + canvas — LE fichier critique)
│   │   └── yolo_inference.dart      (load model + preprocess + infer + postprocess)
│   ├── widgets/bbox_overlay.dart    (CustomPainter)
│   └── screens/
│       ├── camera_screen.dart
│       └── result_screen.dart
└── test/
    ├── nms_test.dart
    ├── coordinate_mapping_test.dart
    └── yolo_postprocess_test.dart
```

## Notes de design importantes

- **Pas de l10n v1**: strings FR hardcodés. `flutter_localizations` ajouté en v1.x
  quand on voudra EN + autres langues.
- **Pas d'historique v1**: capture → inférence → affichage → retour caméra.
  Room/sqflite en v1.x.
- **XNNPACK CPU only**: pas de NNAPI (trop buggy sur vieux SoC), pas de GPU delegate.
- **Android uniquement**: pas d'iOS, part de marché négligeable chez cible.
- **minSdk 24** (Android 7): seuil le plus bas que `tflite_flutter` supporte.
