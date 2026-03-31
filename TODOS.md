# TODOS

## Inference timeout (v1.x)
**What:** Add a 5-second timeout to TFLite inference with a "Still analyzing..." indicator.
**Why:** Without it, the app silently hangs on very slow hardware. The UX is bad — the consultant doesn't know if the app is working or frozen.
**Pros:** Simple fix (~10 lines), prevents worst-case UX failure.
**Cons:** None significant.
**Context:** MobileNetV3-Small typically runs in <100ms, so this fires rarely. But on the oldest supported devices (Snapdragon 410 era, Android 7), inference can take longer. A timeout with a loading indicator is standard UX practice.
**Where to start:** `InferenceService.classify()` — wrap the `tflite_flutter` `Interpreter.run()` call with a `Future.timeout(Duration(seconds: 5))`.

---

## v2 Object Detection (post v1 validation)
**What:** Replace the MobileNetV3-Small classifier with a YOLOv8-nano object detector that draws bounding boxes around disease regions.
**Why:** The classifier works well for single-pod closeups. If consultants photograph tree branches with multiple pods, detection adds localization — "this specific pod on the left has black pod rot."
**Pros:** More informative output, handles multi-pod photos, better for training farmers visually.
**Cons:** Larger model (~6MB vs ~3MB), harder to train (needs bounding box annotations in training data, not just class labels), slower inference.
**Context:** KaraAgroAI and Sykes datasets have classification labels, not bounding boxes. Building the detector requires either re-labeling with bounding boxes or using a separate annotated dataset. Defer until v1 proves the classifier is useful in the field.
**Where to start:** YOLOv8-nano + TFLite export guide. tflite_flutter supports detection models.
**Blocked by:** v1 shipped and validated with consultant in field.

---

## French localization (v1.x or v2)
**What:** Add French localization to the app — disease names, treatment recommendations, UI labels, and error messages.
**Why:** The consultant client works in a francophone West African context (Côte d'Ivoire, possibly others). English treatment recommendations reduce trust and usability.
**Pros:** Directly improves v1 usability for the current client. Essential for any farmer-facing v2 in francophone countries.
**Cons:** All recommendation text must be translated and reviewed by an agronomist — not just machine-translated.
**Context:** Flutter's `l10n` (flutter_localizations + intl) is the standard approach. Treatment recommendation text should be reviewed by the consultant client before shipping.
**Where to start:** Add `flutter_localizations` dependency, create `lib/l10n/app_fr.arb` alongside `app_en.arb`. Get consultant to review the French agronomic text.
**Blocked by:** Clarifying call with consultant (ask which language they prefer for v1).
