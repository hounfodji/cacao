# TODOS

Scope v1 = détection (YOLOv8n + 4 classes). Target user = petit producteur
ouest-africain, langue FR. Voir `cacao_research.md` + design doc 2026-04-07
(`~/.gstack/projects/cacao/hounfodji-master-design-20260407-084624.md`) pour
le contexte.

---

## E2E bbox sanity sur device Android (v1 — BLOQUANT)
**What:** Déployer l'app sur un vrai téléphone Android, photographier une
cabosse (ou image imprimée de cabosse noire), vérifier que le bbox apparaît
visuellement sur la cabosse et pas dans un coin aléatoire.
**Why:** C'est le seul gate du design doc qui ne peut pas être testé en Python
ou en `flutter test`. Le risque est le xy inversion silencieux entre l'espace
du modèle (TF HWC) et le Canvas Flutter. Si ce check échoue, l'app est cassée
pour l'utilisateur même si tous les mAP sont bons.
**Where to start:** `cd app && flutter run --release`, puis dérouler les 8
sanity checks du plan `.claude/plans/robust-sleeping-whistle.md`.

---

## Inference timeout (v1.x)
**What:** Ajouter un timeout 5s sur l'appel `interpreter.run()` avec un
indicateur "Analyse en cours...".
**Why:** Sur très vieux SoC (Snapdragon 410 era), l'inférence peut dépasser 1s.
Sans timeout, l'app semble figée.
**Pros:** ~10 lignes, UX safer.
**Where to start:** `lib/services/yolo_inference.dart` → wrap `interpreter.run()`
dans `Future.timeout(const Duration(seconds: 5))`.

---

## Historique local (v1.x)
**What:** Sauvegarder chaque diagnostic (photo + detections + date) dans une
base SQLite locale. Écran "Historique" accessible depuis la caméra.
**Why:** Utile pour le producteur qui veut suivre l'évolution d'un arbre,
et pour nous (debugging field) si on récupère les devices plus tard.
**Pros:** Améliore la valeur perçue de l'app.
**Cons:** Ajoute `sqflite`, `path_provider`, une couche repository.
**Where to start:** `sqflite` + `DiagnosisRepository` + nouvel écran `HistoryScreen`.

---

## Localization EN (+ autres) (v1.x ou v2)
**What:** Ajouter `flutter_localizations` + `intl`, extraire tous les strings
FR hardcodés dans `app_fr.arb`, ajouter `app_en.arb`.
**Why:** v1 est FR-only par simplicité. Ouvrir l'anglais élargit le marché
(Ghana, Nigeria francophones vs anglophones).
**Cons:** Le texte de conseil agricole (`advice_fr.json`) doit être traduit
par un agronome, pas par machine.
**Where to start:** `flutter gen-l10n`, déplacer les strings français du code
vers `app_fr.arb`.

---

## Révision agronomique du texte conseil (v1.x, bloquant pour le déploiement réel)
**What:** Faire relire `assets/advice/advice_fr.json` par un agronome cacao
ouest-africain (CNRA, CRIG, ou vulgarisateur local).
**Why:** Le texte actuel est un stub écrit de mémoire — recommander un
fongicide cuivré sans avis local peut être contre-productif ou dangereux.
**Where to start:** Contacter CNRA (Côte d'Ivoire) ou CRIG (Ghana).

---

## Contact vulgarisateur intégré (v2)
**What:** Écran "Appeler un vulgarisateur" qui lance l'appareil téléphone
(numéros pré-remplis par région).
**Why:** Pont entre le diagnostic et l'action. Le producteur voit CSSVD → il
peut appeler directement le vulgarisateur de sa zone.
**Blocked by:** Partenariat avec le réseau d'extension local.

---

## Severity grading (v2)
**What:** Module séparé pour noter la sévérité de Black Pod Rot (dataset
`bpr_levels` Davao, 5 niveaux).
**Why:** Un BPR niveau 1 vs niveau 5 a des actions très différentes.
**Blocked by:** v1 ship + retour terrain.

---

## Classes v2 (`pod_borer`, `mirid`, `monilia`)
**What:** Ajouter les classes différées du design doc v1.
**Blocked by:** collecte de données supplémentaires (mirid n'a aucun dataset
vérifié pour l'instant — à traquer via CRIG ou field collection).
