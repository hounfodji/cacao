# Datasets cacao — catalogue de référence

Ce fichier liste tous les datasets identifiés et vérifiés pour le projet de détection offline de maladies du cacao en Afrique de l'Ouest. Pour le contexte produit et le paysage de recherche, voir `cacao_research.md`.

**Scope v1** (4 classes) : `healthy`, `cssvd`, `anthracnose`, `black_pod`.
**Scope v2** (différé) : `pod_borer`, `mirid`, `monilia`, stades de maturité.

---

## 1. Datasets primaires — Afrique (pertinents v1)

### 1.1 KaraAgroAI Cocoa
- **URL** : https://datasetninja.com/kara-agro-ai-cocoa
- **Source officielle** : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BBGQSP (DOI 10.7910/DVN/BBGQSP)
- **Région** : Ghana (7 régions productrices)
- **Volume** : 11,978 images, 21,712 objets annotés
- **Classes** (8) : `cocoa-swollen-shoot-virus-leaf` (5,638), `anthracnose-leaf` (2,996), `healthy-cocoa-leaf` (2,143), `healthy-cocoa-pod` (842), `anthracnose-pod` (229), `cocoa-swollen-shoot-virus-pod` (116), `healthy-cocoa` (86), `cocoa-swollen-shoot-virus-stem` (58)
- **Format** : bounding boxes (détection)
- **Licence** : CC0 1.0
- **Note v1** : source principale pour `cssvd`, `anthracnose`, `healthy`. Déjà branché dans `ml/01_data_curation.py` via `KARAAGRO_CLASS_MAP`.

### 1.2 CocoaMFDB
- **URL** : https://data.mendeley.com/datasets/9msjjh3np6/2
- **Région** : Côte d'Ivoire (Afrique de l'Ouest — très pertinent)
- **Contenu** : base d'images de cabosses de cacao d'une plantation ivoirienne
- **Note v1** : à intégrer pour augmenter la diversité régionale (West Africa) sur les classes cabosses.

### 1.3 Amini Cocoa Contamination
- **URL** : https://www.kaggle.com/datasets/ohagwucollinspatrick/amini-cocoa-contamination-dataset
- **Origine** : challenge Zindi (Amini Cocoa Contamination Challenge)
- **Région** : Uganda, Tanzania, Ghana, Namibia
- **Volume** : ~12,700 fichiers (10.19 GB)
- **Format** : bounding boxes (multi-maladies par image possible)
- **Licence** : CC BY 4.0
- **Note v1** : utile pour augmenter `healthy` et maladies foliaires en contexte sub-saharien.

### 1.4 Cocoa Pod Maturity Stages (Ghana)
- **URL** : https://www.kaggle.com/datasets/aidooben/cocoa-pod-maturity-stages-dataset
- **Région** : Central Region, Ghana
- **Volume** : 3,472 images (15.56 GB)
- **Classes** (6) : `young`, `immature`, `mature-unripe`, `ripe`, `overripe`, `spoilt`
- **Format** : YOLO bounding boxes (annotations MakeSense.AI)
- **Licence** : CC BY 4.0
- **Note** : hors scope v1 (maturité ≠ maladie). Réservé v2 pour module estimation rendement / maturité.

---

## 2. Datasets secondaires — autres régions (augmentation)

### 2.1 CocoaNet / Sykes Ecuador
- **URL** : https://osf.io/2fw6g/overview
- **Code associé** : https://github.com/jrsykes/CocoaReader
- **Région** : Ecuador + images scrapées web
- **Volume** : ~7,000 images Ecuador (compressées 500×500), organisées en `Easy` / `Difficult` / `Unsure` (semi-supervisé)
- **Classes** : BPR (Black Pod Rot), FPR (Frosty Pod Rot), WBD (Witches' Broom), Healthy, NotCocoa
- **Note v1** : source principale pour `black_pod` (via dossier BPR). Déjà branché dans `ml/01_data_curation.py` via `SYKES_CLASS_MAP`.

### 2.2 CocoaMoniliaSet
- **URL** : https://zenodo.org/records/17716661
- **Volume** : 1,953 images
- **Classes** (4) : `h0` (healthy), `m1` (humps, cycle 1), `m2` (oily/brown spots, cycles 2+3), `m3` (white powder/sporulation, cycle 4)
- **Format** : COCO 1.0, YOLO, masques de segmentation 1.1 (annotations polygonales CVAT)
- **Note** : hors scope v1 (Monilia = maladie Amérique latine). Réservé v2 si extension géographique.

### 2.3 Cocoa Disease Detection YOLOv8n (combined)
- **URL** : https://www.kaggle.com/datasets/collinsadekoye/cocoa-disease-detection-yolov8n
- **Volume** : ~13,200 fichiers (11.7 GB)
- **Classes** (5) : `Anthracnose`, `CSSVD`, `Phytophthora`, `Monilia`, `Healthy`
- **Licence** : CC0
- **Note v1** : intéressant car combine plusieurs sources et cible déjà déploiement edge Android (yolov8n). À évaluer pour augmentation `cssvd` / `anthracnose`.

### 2.4 Black Pod Rot & Pod Borer
- **URL** : https://www.kaggle.com/datasets/kenfackbruno/black-pod-rot-and-pod-borer-on-cocoa-pod
- **Volume** : 4,878 fichiers (850 MB)
- **Format** : COCO + masques
- **Note v1** : utile pour `black_pod` ; pertinent v2 pour `pod_borer`.

### 2.5 Cocoa Diseases Localization (Pérou)
- **URL** : https://www.kaggle.com/datasets/bryandarquea/cocoa-diseases
- **Volume** : 3,113 images
- **Classes** (5) : Healthy (2,125), Corncob disease / *Carmenta foraseminis* (217), Witches' broom (94), Moniliasis (351), Phytophthora (328)
- **Format** : YOLO
- **Licence** : CC BY-NC-SA 4.0 (attention non-commercial)
- **Note v1** : `Phytophthora` ≈ proxy `black_pod`.

### 2.6 Cacao Diseases (Davao, Philippines)
- **URL** : https://www.kaggle.com/datasets/zaldyjr/cacao-diseases
- **Volume** : ~4,390 images (1080×1080)
- **Classes** (3) : Black Pod Rot, Healthy, Pod Borer
- **Licence** : CC BY-SA 4.0
- **Note v1** : utile pour `black_pod` et `healthy` ; v2 pour `pod_borer`.

### 2.7 Black Pod Rot Levels
- **URL** : https://www.kaggle.com/datasets/zaldyjr/black-pod-rot-levels
- **Volume** : ~939 images (1080×1080)
- **Classes** : 5 niveaux d'infection (Level 1 → Level 5)
- **Licence** : CC BY-SA 4.0
- **Note** : utile pour module futur de gradation de sévérité (pas classification binaire).

### 2.8 Cocoa Diseases (YOLOv4)
- **URL** : https://www.kaggle.com/datasets/serranosebas/enfermedades-cacao-yolov4
- **Volume** : 627 images (Fito 215, Monilia 211, Sana 201)
- **Classes** (3) : Fito (Phytophthora), Monilia, Sana (Healthy)
- **Note v1** : petit dataset, complément Phytophthora → `black_pod`.

---

## 3. Datasets post-récolte (hors scope v1)

### 3.1 Cocoa Beans Image Dataset
- **URL** : https://www.kaggle.com/datasets/khawaritzmiabdallah/cocoa-beans-image-dataset
- **Volume** : 614 fichiers (7.17 MB)
- **Classes** (6) : `Bean_Fraction`, `Broken_Beans`, `Fermented`, `Moldy`, `Unfermented`, `Whole_Beans`
- **Citation** : Adhitya et al., *Agronomy* 2020, 10, 1642
- **Note** : qualité des fèves post-récolte — hors scope détection maladies au champ.

---

## 4. Mapping vers les classes v1

| Classe v1     | Sources principales (déjà branchées)        | Sources d'augmentation possibles                                              |
|---------------|---------------------------------------------|-------------------------------------------------------------------------------|
| `healthy`     | KaraAgroAI (`healthy-cocoa*`), Sykes (Healthy) | CocoaMFDB, Amini, YOLOv8n combined, Davao, Cocoa Localization                |
| `cssvd`       | KaraAgroAI (`cocoa-swollen-shoot-virus-*`)  | YOLOv8n combined                                                              |
| `anthracnose` | KaraAgroAI (`anthracnose-*`)                | YOLOv8n combined                                                              |
| `black_pod`   | Sykes (BPR)                                 | CocoaMFDB, Kenfack BPR/Pod Borer, Cocoa Localization (Phytophthora), Davao, YOLOv4 (Fito), Black Pod Rot Levels |

**Pipeline v1** : seuls KaraAgroAI et Sykes sont consommés par `ml/01_data_curation.py` (`KARAAGRO_CLASS_MAP`, `SYKES_CLASS_MAP`). Les autres sources sont des candidats d'augmentation à évaluer après le premier baseline.

**Différé v2** : `pod_borer` (Davao, Kenfack), `mirid` (aucune source vérifiée — à rechercher), `monilia` (CocoaMoniliaSet, Cocoa Localization), maturité (Ghana Maturity Stages).
