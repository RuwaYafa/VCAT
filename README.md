
# 😌 VCAT – Video Content Analysis Tool for Digital Forensics

**VCAT** (Video Content Analysis Tool) is a digital forensic solution designed to assist analysts and law enforcement in processing and analyzing video evidence using AI-powered modules. It integrates object detection, OCR, and speech recognition into one streamlined pipeline.

---

## Key Features

- **Object Detection** via GroundingDINO – Detects visual targets in video frames using text-based prompts.
- **Optical Character Recognition (OCR)** via PaddleOCR – Extracts text from video frames (e.g., signs, billboards).
- **Speech Recognition** via Whisper – Transcribes spoken content in video audio.
- **AI Prompt Search** – Use natural language prompts to focus search on specific events or objects.
- **Court-Admissible Reporting** – Structured output with timestamps, GPS, confidence score, and hash verification.
- Built on Google Colab for easy deployment and reproducibility.

---

## Workflow Overview

<img width="768" alt="Flowchart drawio" src="https://github.com/user-attachments/assets/94fa8ee8-d718-4bad-a406-3db8a35de34f" />

1. **Input**
   - Case info, forensic image, analyst data
   - Keywords/prompts like: `"billboard, white car, #StopTheGenocide, Palestine"`
2. **Video Processing**
   - Extract frames and audio
   - Run through AI modules (GroundingDINO, PaddleOCR, Whisper)
3. **Filtering & Grouping**
   - Based on frame index and confidence score
4. **Output Report**
   - Includes parsed results: timestamps, artifacts, confidence, GPS
   - Format aligned with forensic standards (NIST, SWGDE)

---

## Sample UI

<img width="561" alt="VCATUI" src="https://github.com/user-attachments/assets/5823d597-b298-4050-8e87-f58f6013fed2" />

- Google Colab form interface
- Input fields for paths, case info, and prompts
- Checkboxes to activate modules
- One-click report generation

---

## Input & Output

### Input
- Video files (from forensic image)
- Analyst and case metadata
- Prompt keywords for search focus

### Output
- JSON/CSV structured result files
- Auto-generated forensic report via ReportLab
- Confidence-ranked evidence per frame/audio

---

## Forensic Integrity

- All hashes are calculated on video files to ensure authenticity.
- Adheres to **NIST 800-86** and **SWGDE** guidelines.
- Supports chain-of-custody documentation.

---

## Thesis & Research

This tool is part of the master's thesis:

> **“A Proposed Digital Forensic Tool for Video Content Analysis in the Investigation Process”**  
> *by Ruwa’ Fayeq Suleiman Abu Hweidi – PTUK, 2024*  
> [V-CATMethod.drawio.pdf](https://github.com/user-attachments/files/21107721/V-CATMethod.drawio.pdf)

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
