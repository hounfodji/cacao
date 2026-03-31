# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **offline Android cacao disease detection app** for West African smallholder farmers. The project is currently in the research/planning stage. The core finding: no production-quality, offline AI cacao disease app exists for West Africa — DR CACAO (the only Play Store app) targets Latin American diseases and requires internet.

The full research landscape analysis is in `cacao_research.md`. Read it before making architectural decisions.

## Key Technical Decisions (from research)

**Target platform**: Android, offline-first, low-end devices (2–3GB RAM, 16–32GB storage, ARM CPU, no GPU/NPU guarantee).

**ML framework**: TensorFlow Lite (not ONNX Runtime or PyTorch Mobile) — best ARM optimization, widest Android coverage, proven by PlantVillage Nuru.

**Recommended models**:
- Classification: MobileNetV3-Small (transfer learning from ImageNet) — ~3MB, <100ms on Snapdragon 450
- Detection: YOLOv8-nano — ~3.1MB FP16, 87.4% mAP@0.5, 0.61s inference on edge hardware

**Primary dataset**: KaraAgroAI (17,703 images, sub-Saharan Africa context, CSSVD + pod diseases) — most geographically relevant. Secondary: Sykes Ecuador dataset (7,220 images, strong benchmark).

**Target diseases** (West Africa priority): Black pod rot (*Phytophthora palmivora/megakarya*), CSSVD (Cocoa Swollen Shoot Virus Disease), pod borer, mirid bugs. NOT monilia/witches' broom (Latin America diseases).

**Reference architecture**: Philippines Offline Cacao App (arXiv:2602.00216) — 96.93% validation accuracy, 84.2% field expert agreement, fully offline Android. PlantVillage Nuru is the gold standard for offline crop AI in Africa (SSD MobileNet + TFLite).

## Development Tools

This project uses the [gstack](https://github.com/garrytan/gstack) AI engineering framework installed at `.claude/skills/gstack/`. Key skills available:

- `/plan-eng-review` — Architecture review before building
- `/qa` — Browser/UI testing with real Chromium
- `/ship` — Test, version bump, create PR
- `/investigate` — Systematic root-cause debugging
- `/review` — Pre-landing code review

See `.claude/skills/SKILL.md` for the full list.

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
