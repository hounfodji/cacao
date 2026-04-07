# AI-powered cacao disease detection: a complete landscape analysis

**No production-quality, offline-capable AI app exists for detecting cacao diseases in West Africa — creating an enormous opportunity.** Despite 30+ published research papers, 10+ public datasets, and several research prototypes, the gap between academic work and a farmer-ready Android tool remains wide. The only app store listing, DR CACAO, targets Latin American diseases and requires connectivity. This report maps every known resource — apps, models, datasets, papers, institutional initiatives, and deployment frameworks — to provide a complete blueprint for building an offline Android cacao disease detector for West African smallholders.

---

## The competitive landscape is fragmented and underdeveloped

### Cacao-specific mobile apps

Only **one production app** currently exists on Google Play: **DR CACAO** (by Agrinapsis), which detects moniliasis, black pod, and witches' broom — diseases relevant to Latin America, not West Africa. It appears to require internet connectivity and discloses no accuracy metrics.

Seven research prototypes have been published but none are publicly downloadable:

- **Cocoa Companion** (Penn State/Ghana, 2022): SSD MobileNetV2, cloud-based, >80% confidence for swollen shoot and black pod. Not offline.
- **Philippines Offline Cacao App** (arXiv:2602.00216, 2026): The most architecturally relevant reference — a deep learning model running offline on Android achieving **96.93% validation accuracy** and **84.2% agreement with expert technicians** in field tests. Detects pod borer, mirid bug, black pod rot, and vascular dieback.
- **EfficientDet-Lite4 Android App** (Ecuador, arXiv:2401.01247, 2024): Native Android app for monilia and black pod detection with **0.83 mAP**. Designed for on-device inference.
- **SSD MobileNetV2 FPN-Lite** (Philippines, 2025): Lightweight mobile architecture achieving **84.62% mAP** for monilia and phytophthora.
- **Flutter-based Cacao System** (Philippines, 2025): Cross-platform app with **93.7% classification accuracy** and treatment recommendations.
- **AuToDiDAC** (Philippines, 2017): Older SVM-based approach for black pod severity assessment at **84% pixel-level accuracy**.
- **Basri et al.** (Indonesia, 2020): Android Java app for pest/disease classification with 615 images.

**Critical insight: Mars Wrigley, ICRAF, WCF, and all major chocolate companies have invested zero in publicly available AI disease detection apps.** Their digital investments focus on traceability, supply chain mapping, and farmer advisory tools — not on-device disease AI.

### Competitor apps worth studying as architectural references

**PlantVillage Nuru** is the gold standard for offline crop disease AI in Africa. Built for cassava (CMD, CBSD) in East Africa, it runs entirely offline on low-end Android phones using **SSD MobileNet + TensorFlow Lite**. It achieved **93–98% accuracy** in controlled settings, though field performance drops significantly for mild symptoms. Nuru's architecture — transfer learning from COCO, Fritz AI for model management, offline-first design — is the closest template for a cacao equivalent.

**TumainiAI** (Alliance of Bioversity/CIAT) detects 7 banana diseases with **90% average success** across 6 countries, supports **offline mode since v2**, and runs on TensorFlow with ResNet50/InceptionV2 backends. It offers French, Spanish, and Swahili language support, making it directly relevant to francophone West Africa.

**Plantix** (PEAT GmbH) covers **780+ plant damages** across 30+ crops with 10M+ downloads, but is entirely cloud-dependent. **Agrio** similarly requires internet. Neither works for offline scenarios.

---

## Research has evolved from basic CNNs to edge-deployable YOLO models

Across **30+ published papers** (2019–2026), the field has progressed through four distinct phases.

**Phase 1 (2019–2020)** relied on classical ML. Coulibaly et al. (2020) from Côte d'Ivoire used a custom CNN on leaf images for swollen shoot detection, achieving 84% accuracy on ~5,000 images — one of the first papers from an African cocoa-producing nation. Datasets were tiny (50–615 images).

**Phase 2 (2021–2022)** introduced transfer learning. Kumi et al. (2022) published "Cocoa Companion," comparing four CNN-based detectors — **SSD MobileNetV2 won** with >80% confidence for swollen shoot and black pod. Montesino et al. (2021) applied ResNet18 for *Phytophthora palmivora* detection, reaching **83% disease accuracy** on 1,596 images. The Colombia **Cocoa Diseases (YOLOv4) dataset** (312 images, 3 classes) emerged as the community's most-reused benchmark.

**Phase 3 (2023–2024)** saw architecture diversification. Ferraris & Sartor et al. (2023) deployed **YOLOv5m for Côte d'Ivoire cocoa** achieving 98% healthy fruit detection but ~30% confusion between disease types. Sykes et al. introduced **PhytNet** (2024), a custom ultra-lightweight CNN at just **1.19 GFLOPS** — designed specifically for plant pathology on small datasets, with infrared imaging revealing that the most informative spectra for disease detection lie **outside the visible spectrum**. A pivotal paper on CSSVD detection (arXiv:2405.04535) used the **KaraAgroAI dataset** (17,703 images from sub-Saharan Africa) with ResNet50 achieving **95.39% precision** and **94% accuracy**.

**Phase 4 (2025–2026)** brought ensemble methods and deployment focus. An ensemble combining VGG16/19 + ResNet50/101 + InceptionV3 + Xception with bagging achieved **100% test accuracy** on 6,000 images (though on a simple 3-class problem). YOLOv8n on cocoa pods yielded **87.4% mAP@0.5** with **0.61s inference** — suitable for real-time edge deployment. Sykes et al. (2025) demonstrated that **ResNet18 + semi-supervised learning + dynamic focal loss** was the strongest contender for real-world deployment, publishing a high-quality **7,220-image benchmark dataset** on OSF.

Two systematic reviews now consolidate the field: Sykes et al. (2023) in *Applications in Plant Sciences* and Alvarado et al. (2025) in *Agriculture* (MDPI), the latter covering 88 studies from 2019–2024.

---

## Available datasets range from 312 to 17,703 images

| Dataset | Images | Classes | Format | Source |
|---|---|---|---|---|
| **KaraAgroAI Cocoa** | 17,703 | 3 (Healthy, CSSVD, Anthracnose) | Classification | Harvard Dataverse (Ghana) |
| **Sykes Ecuador Dataset** | 7,220 | 5 (BPR, FPR, WBD, Healthy, NotCocoa) | Classification | OSF (osf.io/2fw6g) |
| **Kaggle cacao-diseases** (zaldyjr) | ~4,390 | 3 (Healthy, Black Pod, Pod Borer) | Classification | kaggle.com/datasets/zaldyjr/cacao-diseases |
| **Roboflow Miss Nyarko** | ~3,009 | 4 (Black Pod, Frosty Pod, Healthy, Mirid) | YOLO detection | Roboflow Universe |
| **CocoaMoniliaDataSet** | 1,953 | 4 severity stages | COCO/YOLO/Segmentation | ScienceDirect (2025) |
| **Cocoa Diseases YOLOv4** (Colombia) | 312 / 1,591 objects | 3 (Healthy, Phytophthora, Monilia) | YOLOv4 bbox | Kaggle (serranosebas) |
| **Amini/Zindi Competition** | ~5,500 | Multiple leaf diseases | Detection | Zindi/Makerere AI Lab |

**The KaraAgroAI dataset is the most directly relevant for West Africa**, containing images from sub-Saharan African cocoa farms with CSSVD — the disease devastating Ghana and Côte d'Ivoire. The Sykes Ecuador dataset offers the highest-quality, multi-camera field images but covers Latin American diseases (BPR, FPR, WBD). **No large-scale, labeled dataset exists specifically for West African cacao pod diseases** combining black pod (*P. megakarya*, the dominant West African species), CSSVD leaf symptoms, and mirid damage — building one will be essential.

---

## Open-source code and pretrained weights are available but sparse

The **CocoaReader** repository (github.com/jrsykes/CocoaReader, MIT license) is the most complete research codebase. It includes PhytNet and ResNet18 implementations in PyTorch, supports quantization (`--quantise` flag), provides pretrained `.pkl` weights on OSF, and covers BPR, FPR, WBD, and healthy classification. PhytNet's **1.19 GFLOPS** make it exceptionally mobile-friendly.

The **Amini/Zindi competition solutions** offer production-ready object detection code: the 3rd-place solution uses **YOLO11s** with stratified K-fold and Weighted Box Fusion ensemble (github.com/koleshjr), while the 2nd-place uses DINO + Swin-B (github.com/stefan027) — heavier but more accurate.

**Roboflow** hosts several cocoa disease datasets with pre-trained YOLOv5/v8 models exportable directly to TFLite, ONNX, and CoreML — the fastest path from zero to a working mobile model. The **Wolcan S** and **data** workspaces provide ready-to-deploy models.

No dedicated cocoa disease models exist on **HuggingFace** or **TensorFlow Hub**. Google's **CropNet** provides cassava disease classifiers as TFLite models and serves as an excellent pipeline reference via the official on-device tutorial.

---

## TensorFlow Lite is the clear choice for offline Android deployment

After comparing four frameworks across inference speed, model size, memory usage, and low-end device compatibility, **TensorFlow Lite (LiteRT)** emerges as the optimal runtime for this use case.

| Criterion | TFLite | ExecuTorch | ONNX Runtime | MediaPipe |
|---|---|---|---|---|
| **ARM inference** | ~23ms ⭐ | ~38ms | ~31ms | ~23ms (wraps TFLite) |
| **Model size (INT8)** | Smallest ⭐ | 3× larger | Moderate | Same as TFLite |
| **Memory usage** | ~89MB ⭐ | ~126MB | ~112MB | ~89MB |
| **Low-end device support** | Excellent ⭐ | Fair | Fair | Excellent |
| **Documentation maturity** | Best ⭐ | Growing | Good | Good |

For the fastest development path, **MediaPipe Tasks** (which wraps TFLite) provides the highest-level API with built-in camera integration, automatic preprocessing, and the Model Maker library for transfer learning with as few as 100 images per class. For maximum control, use the raw TFLite interpreter with CameraX.

### Recommended model architecture for the app

> **2026-04-07 update — data-driven decision:** v1 is now **YOLOv8n object detection**, not MobileNetV3 classification. After downloading all 13 datasets (~53 GB), 10 of 13 are in YOLO/Supervisely/COCO bbox format with ~50,000 annotated objects. KaraAgroAI (the primary West Africa source) is bbox, not folder-classes — the original assumption was wrong. Forcing the data into classification would require a lossy bbox-cropping pipeline. Both MobileNetV3 and YOLOv8n run 100% offline via TFLite — "offline" was a false constraint on model choice. Detection also gives a better field UX: the farmer sees a bbox over the diseased area instead of a single image-level label. See `~/.gstack/projects/cacao/hounfodji-master-design-20260407-084624.md` for the full data audit and architectural decision.

**v1 model — object detection: YOLOv8n.** Achieves **87.4% mAP@0.5** on cocoa pods with **0.61s inference** on edge hardware, exports directly to TFLite via Ultralytics (`model.export(format='tflite', half=True)`), ~3 MB FP16 — same Android footprint as MobileNetV3-Small INT8. Trained via the official `ultralytics` package with native TFLite export.

**Alternative (deferred):** MobileNetV3-Small classification (~6 MB float32, ~2 MB INT8, 94–99% on plant disease benchmarks) remains the right choice if the dataset becomes folder-class dominant in v2 or if a per-class confidence head is added on top of detection. Not the v1 path.

### Critical deployment specifications for West African phones

Target devices like Tecno, Infinix, and Samsung Galaxy A series typically have **2–3GB RAM, 16–32GB storage, ARM Cortex-A CPUs**, and run **Android 7.0+**. The model file should stay **under 5MB** (INT8 quantized), the total app under **30MB**, and inference under **100ms**. Use the **XNNPACK delegate** (CPU-optimized, not GPU) to preserve battery life. Use CameraX with single-capture mode rather than continuous analysis — a farmer photographs a pod, then receives the diagnosis.

### Training-to-deployment pipeline

Train in TensorFlow/Keras (or PyTorch via `ai_edge_torch` converter) → apply INT8 post-training quantization with a representative calibration dataset → embed TFLite metadata (labels, normalization parameters) → place `.tflite` file in Android `assets/` → integrate with MediaPipe Tasks ImageClassifier or TFLite Task Library → use CameraX for camera integration → store classification history in Room database → use WorkManager + Firebase ML for opportunistic model updates when connectivity is available.

---

## Institutional investment is massive but not in disease AI

**Mars Inc.** has invested **$1 billion** (2018–2028) in "Cocoa for Generations" but focuses on genomics, traceability, and a **SwissDeCode DNAFoil rapid diagnostic kit** for CSSVD (detects virus in asymptomatic leaves in <60 minutes). Mars opened the **Cocoa Advanced Research Laboratory (CARL)** in Indonesia in 2024 for IPM research. **Barry Callebaut** mapped **181,861+ farms** with GPS polygons and partnered with Microsoft for digital transformation. **Cargill** deployed digital cooperative management systems covering **117,111 farmers** through barcoded traceability.

Among research institutions, **CIRAD** developed and compared Faster-RCNN, Mask-RCNN, and YOLOv3 for cocoa pod detection on **794 images from Côte d'Ivoire and Cameroon**, achieving 74% accuracy with Faster-RCNN. **CRIG** (Ghana) has explicitly called for AI and image acquisition tools for CSSVD management but has not yet developed one. **CNRA** (Côte d'Ivoire) participates in the DNAFoil CSSVD diagnostic kit but has no AI app.

**GIZ's "DigiGreen & Agri" project** (2024+) in partnership with the EU and Orange is actively training 2,000 youth and supporting 20 startups in digital cocoa innovation in Côte d'Ivoire — potentially a route for funding and distribution of an offline disease detection app.

---

## A West Africa–specific disease detection app must address these diseases

The disease profile for West Africa differs fundamentally from Latin America:

- **Black pod disease** (*Phytophthora megakarya* in West Africa, more aggressive than *P. palmivora*): causes **23% of global crop losses**, dominant across all West African countries
- **Cocoa Swollen Shoot Virus Disease (CSSVD)**: transmitted by mealybugs, **780,000+ hectares infected** in Ghana and Côte d'Ivoire, can kill trees within 2–3 years
- **Mirid/capsid bug damage**: major insect pest across West Africa
- **Anthracnose**: increasingly documented alongside CSSVD in KaraAgroAI dataset

Note that **moniliasis (frosty pod rot) and witches' broom are NOT present in West Africa** — they are exclusively Latin American diseases. Most existing models and datasets are biased toward these diseases, creating a significant domain gap that must be addressed through custom West African data collection.

---

## Actionable blueprint for building the app

**Phase 1 — Data collection and baseline model (months 1–3):** Collect 2,000+ field images across target diseases (black pod, CSSVD leaves, mirid damage, healthy pods/leaves) using multiple smartphone models across several farms. Supplement with the KaraAgroAI dataset (17,703 images) and Kaggle cacao-diseases (4,390 images). Use Roboflow for annotation and augmentation.

**Phase 2 — Model training and optimization (months 2–4):** Train MobileNetV3-Small with transfer learning from ImageNet. Apply Sykes et al.'s semi-supervised learning approach with dynamic focal loss to maximize generalization on small datasets. Include a "not-cacao" class following their finding that this dramatically improves robustness. Apply INT8 quantization and validate on a held-out test set.

**Phase 3 — Android integration (months 3–5):** Build with Kotlin + CameraX + MediaPipe Tasks. Bundle the quantized TFLite model (~2–5MB) in the APK. Implement offline-first architecture with Room for local history and WorkManager for opportunistic model updates. Add French language UI. Include treatment recommendations and extension service contacts.

**Phase 4 — Field validation (months 5–7):** Test on Tecno/Infinix/Samsung A-series phones in actual cocoa farms. Measure the lab-to-field accuracy gap (expect 10–20% drop based on Ramcharan et al. 2019). Iterate on data collection targeting failure cases.

**Potential partners for scale:** CRIG/COCOBOD (Ghana), CNRA (Côte d'Ivoire), GIZ DigiGreen & Agri program, World Cocoa Foundation TechXchange, Mars Cocoa for Generations, and the Amini/Zindi AI competition community.

---

## Conclusion

The cacao disease detection space sits at an inflection point: the scientific foundations are solid (30+ papers, improving architectures, growing datasets), the deployment technology is mature (TFLite, MobileNetV3, CameraX), and institutional demand is explicit (CRIG calling for AI tools, GIZ funding digital cocoa startups). Yet **no one has built the product**. The three most actionable starting points are: (1) the CocoaReader codebase with PhytNet/ResNet18 and quantization support, (2) the KaraAgroAI dataset as the only large-scale sub-Saharan African cocoa image collection, and (3) the PlantVillage Nuru architecture as a proven template for offline AI on low-end African smartphones. The single largest risk is the **dataset gap for West African–specific diseases** — particularly *P. megakarya* black pod and CSSVD on leaves — which will require dedicated field collection partnerships with CRIG or CNRA to overcome.












liens

Voici la liste complète des liens et sources utilisés dans le rapport :

**Apps mobiles**

1. **DR CACAO** (Google Play) — https://play.google.com/store/apps/details?id=com.agrinapsis.drcacao&hl=en
2. **PlantVillage Nuru** (App Store) — https://apps.apple.com/us/app/plantvillage-nuru/id1441395371
3. **Plantix** (Google Play) — https://play.google.com/store/apps/details?id=com.peat.GartenBank&hl=en_IN&gl=US
4. **Agrio** — https://agrio.en.aptoide.com/app
5. **TumainiAI** (CGIAR) — https://www.cgiar.org/innovations/tumaini-an-ai-powered-mobile-app-for-pests-and-diseases
6. **TumainiAI** (Alliance Bioversity-CIAT) — https://alliancebioversityciat.org/tools-innovations/tumaini-app-crop-health-bananas-beans

**Articles scientifiques**

7. **Deep Learning-Based Computational Model for Disease Identification in Cocoa Pods** (arXiv 2024) — https://arxiv.org/abs/2401.01247
8. **Development of a Cacao Disease Identification and Management App** (arXiv 2026) — https://arxiv.org/abs/2602.00216
9. **Cocoa Companion** (ScienceDirect) — https://www.sciencedirect.com/science/article/pii/S1877050922006184
10. **Cocoa Companion** (ResearchGate) — https://www.researchgate.net/publication/362662955_Cocoa_Companion_Deep_Learning-Based_Smartphone_Application_for_Cocoa_Disease_Detection
11. **Cocoa Companion** (Penn State) — https://pure.psu.edu/en/publications/cocoa-companion-deep-learning-based-smartphone-application-for-co
12. **Efficient detection of cacao pod diseases using SSD** (WJARR 2025) — https://journalwjarr.com/sites/default/files/fulltext_pdf/WJARR-2025-0128.pdf
13. **A Novel AI-Driven System for Cacao Disease Classification** (ResearchGate) — https://www.researchgate.net/publication/391970784_A_Novel_AI-Driven_System_and_Method_for_Cacao_Variety_and_Disease_Classification_and_Treatment_Recommendation_Using_Image_Analysis
14. **AuToDiDAC: Automated Tool for Disease Detection for Cacao Black Pod Rot** (ScienceDirect) — https://www.sciencedirect.com/science/article/abs/pii/S0261219417302867
15. **Mobile Image Processing Application** (ICICEL 2020) — http://www.icicel.org/ell/contents/2020/10/el-14-10-12.pdf
16. **Deep Learning for Image-Based Cassava Disease Detection** (Frontiers 2017) — https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2017.01852/full
17. **Disease Detection on Cocoa Crops: Systematic Literature Review** (MDPI Agriculture 2025) — https://www.mdpi.com/2077-0472/15/10/1032
18. **Detection of Swollen Shoot Disease in Ivorian Cocoa Trees via CNN** (SCIRP) — https://www.scirp.org/html/2-8103416_98808.htm
19. **Cocoa Pods Diseases Detection by MobileNet Confluence** (IJACSA) — https://thesai.org/Downloads/Volume14No9/Paper_37-Cocoa_Pods_Diseases_Detection_by_MobileNet_Confluence.pdf
20. **Machine Learning as a Strategic Tool for Cocoa Farmers in Côte d'Ivoire** (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10490821/
21. **Machine Learning for Cocoa Farmers in Côte d'Ivoire** (MDPI Sensors) — https://www.mdpi.com/1424-8220/23/17/7632
22. **Improving computer vision for plant pathology** (Wiley/Am. J. Botany) — https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.70010
23. **Image Classification for CSSVD Detection in Cacao Plants** (arXiv 2024) — https://arxiv.org/abs/2405.04535
24. **Enhancing Cocoa Pod Disease Classification via Transfer Learning and Ensemble Methods** (arXiv 2025) — https://arxiv.org/abs/2504.12992
25. **Deep Learning-Based Computer Vision Framework for Early Detection of Cocoa Plant Diseases** (Atlantis Press 2025) — https://www.atlantis-press.com/proceedings/icostas-eas-25/126016927
26. **Epidemiology and Diagnostics of Cacao Swollen Shoot Disease in Ghana** (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10819116/
27. **LDL-MobileNetV3S for potato leaf disease** (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC12585974/
28. **Plant disease detection model for edge computing** (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10748432/
29. **Real-time Image detection of cocoa pods** (CIRAD/Agritrop) — https://agritrop.cirad.fr/607440/

**Datasets**

30. **Cocoa Diseases** (Dataset Ninja) — https://datasetninja.com/cocoa-diseases
31. **KaraAgroAI Cocoa Dataset** — https://air.ug/dataset-details/8/
32. **Cocoa Disease Object Detection** (Roboflow - Wolcan S) — https://universe.roboflow.com/wolcan-s-hd3hj/cocoa-disease
33. **Cocoa Disease Object Detection** (Roboflow - data) — https://universe.roboflow.com/data-yu4yz/cocoa-disease-zl3cl
34. **Amini Cocoa Contamination Challenge** (Zindi) — https://zindi.africa/competitions/amini-cocoa-contamination-challenge/discussions/26931

**Repos GitHub open-source**

35. **CocoaReader** (jrsykes) — https://github.com/jrsykes/CocoaReader
36. **CocoaReader/CocoaNet** — https://github.com/jrsykes/CocoaReader/tree/main/CocoaNet
37. **CocoaReader/CocoaNet/PhytNet_Cocoa** — https://github.com/jrsykes/CocoaReader/tree/main/CocoaNet/PhytNet_Cocoa
38. **PhytNet** (jrsykes) — https://github.com/jrsykes/PhytNet

**Frameworks de déploiement mobile**

39. **CropNet: Cassava Disease Detection** (TensorFlow Hub) — https://www.tensorflow.org/hub/tutorials/cropnet_cassava
40. **MediaPipe Solutions** (Google Developers Blog) — https://developers.googleblog.com/introducing-mediapipe-solutions-for-on-device-machine-learning/
41. **MediaPipe Image Classifier Customization** — https://developers.google.com/mediapipe/solutions/customization/image_classifier
42. **MediaPipe Image Classification for Android** — https://developers.google.com/mediapipe/solutions/vision/image_classifier/android
43. **Google AI Edge Image Classifier Android** — https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/android
44. **MediaPipe Model Maker Guide** — https://developers.googleblog.com/5-things-to-know-before-customizing-your-first-machine-learning-model-with-mediapipe-model-maker/
45. **TFLite vs ONNX Runtime Comparison** — https://samanvya.dev/blog/tflite-vs-onnx-runtime
46. **Optimizing On-Device Inference** (APXML) — https://apxml.com/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/optimizing-on-device-inference
47. **Best Mobile Object Detection Models** (Roboflow Blog) — https://blog.roboflow.com/mobile-object-detection-models/
48. **TFLite + CameraX Tutorial** (Fritz AI) — https://fritz.ai/image-classification-on-android/
49. **TFLite + CameraX** (Medium) — https://medium.com/better-programming/image-classification-on-android-with-tensorflow-lite-and-camerax-4f72e8fdca79
50. **Top 10 Edge AI Frameworks 2025** (Huebits) — https://blog.huebits.in/top-10-edge-ai-frameworks-for-2025-best-tools-for-real-time-on-device-machine-learning/

**Acteurs institutionnels & industrie**

51. **Mars — Cocoa for Generations** — https://www.mars.com/sustainability-plan/cocoa-for-generations
52. **Mars — Cocoa Research & Science** — https://www.mars.com/sustainability-plan/cocoa-for-generations/research-and-science
53. **Mars — Modern Cocoa** — https://www.mars.com/sustainability-plan/cocoa-for-generations/modern
54. **Mars — Segregated Supply Chain** — https://www.mars.com/news-and-stories/press-releases-statements/mars-aims-achieve-segregated-global-cocoa-supply-chain
55. **Barry Callebaut — Forever Chocolate** — https://www.barry-callebaut.com/en-US/sustainability/reporting/our-approach-201920
56. **Barry Callebaut — Strategy** — https://www.barry-callebaut.com/en/about-us/investors/our-strategy
57. **Cargill — Cocoa Sustainability Report 2021** — https://www.cargill.com/doc/1432213708736/cargill-cocoa-sustainability-progress-report-2021.pdf
58. **CRIG / COCOBOD Ghana** — https://cocobod.gh/news/cocoa-research-institute-of-ghana-crig-pioneering-cocoa-sustainability-and-innovation-in-ghana
59. **WCF 2026 Partnership Agenda** — https://sustainablebusinessmagazine.net/agriculture/wcf-reveals-2026-cocoa-partnership-agenda/
60. **GIZ — Cocoa 2.0 Digital** — https://www.giz.de/en/newsroom/news/cocoa-20-digital-answers-climate-change

**Autres références**

61. **PlantVillage in East Africa** (Medium) — https://medium.com/@austin_32493/plantvillage-helping-farmers-in-east-africa-identify-and-treat-plant-disease-9a26b167b400
62. **Plantix & AI in Agriculture** (GSMA) — https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-for-development/programme/agritech/detecting-and-managing-crop-pests-and-diseases-with-ai-insights-from-plantix/
63. **AI Plant Disease Diagnosis** (Farmonaut) — https://farmonaut.com/blogs/ai-plant-app-7-ways-to-diagnose-plant-diseases-fast
64. **Sample diseased cacao pods** (ResearchGate) — https://www.researchgate.net/figure/Sample-images-from-the-data-set-of-diseased-cacao-pods_fig3_305649001
65. **Black pod symptoms** (ResearchGate) — https://www.researchgate.net/figure/Symptoms-of-black-pod-disease-on-immature-cocoa-pod-Photos-taken-by-Oro-Franck_fig2_348171178
66. **AuToDiDAC** (ResearchGate) — https://www.researchgate.net/publication/320363503_AuToDiDAC_Automated_Tool_for_Disease_Detection_and_Assessment_for_Cacao_Black_Pod_Rot
