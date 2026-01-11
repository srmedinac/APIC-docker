# APIC - AI-based Pathology Image Classifier 

A computational pathology pipeline for predicting docetaxel benefit in prostate cancer patients.

## Overview

APIC analyzes **Whole Slide Images (WSI)** from prostate cancer biopsies to generate a **patient-level risk score** that predicts whether a patient with **advanced prostate cancer** will benefit from **docetaxel chemotherapy**.

The pipeline uses AI-driven nuclear segmentation and spatial analysis to extract features from tumor tissue, producing an automated PDF report with:

- Patient-specific risk score
- Clinical interpretation
- Prognostic estimates
- Treatment considerations

### Clinical Context

For patients with localized high-risk or metastatic hormone-sensitive prostate cancer, adding docetaxel to androgen deprivation therapy (ADT) can improve outcomes—but not all patients benefit equally. APIC provides a computational biomarker derived from routine H&E-stained biopsy images to help identify patients most likely to benefit from docetaxel intensification.

---

## Quick Start

### Pull the Docker Image

```bash
docker pull madabhushilabapic/apic:latest
```

### Run the Pipeline (Single Slide)

```bash
docker run --gpus all \
  -v /path/to/your/slides:/data/input_slides:ro \
  -v /path/to/your/output:/data/output \
  madabhushilabapic/apic:latest \
  -i "/data/input_slides/slide_filename.ext" \
  -o /data/output
```

> **Note:** GPU is optional. If unavailable, the pipeline automatically falls back to CPU execution.

See [Run Modes](#run-modes) below for batch processing and multi-slide patient options.

---

## Docker Command Explained

Your data stays on your machine—the container only accesses it through mounted paths.

### Volume Mounts (`-v`)

The `-v` flag maps a folder on **your system** to a folder **inside the container**:

```text
-v /your/local/path:/container/path
    ↑                 ↑
    YOUR SYSTEM       CONTAINER (don't change)
```

| Mount | Your System (change this) | Container Path (keep as-is) |
| ----- | ------------------------- | --------------------------- |
| Input | `/path/to/your/slides` | `/data/input_slides:ro` |
| Output | `/path/to/your/output` | `/data/output` |

### Pipeline Arguments

| Flag | Value | Description |
| ---- | ----- | ----------- |
| `-i` | `/data/input_slides/slide.ext` | Path to WSI **inside container** (use container path + filename) |
| `-o` | `/data/output` | Output directory **inside container** (keep as-is) |

### Optional Docker Flags

| Flag | Description |
| ---- | ----------- |
| `--gpus all` | Enable GPU acceleration (omit if no GPU) |
| `--rm` | Auto-remove container after execution (optional, keeps system clean) |

---

## GPU Support (Optional)

This pipeline supports **GPU acceleration** using the **NVIDIA Container Toolkit**. Follow the steps below to enable GPU support on Ubuntu/Debian systems.

### Install NVIDIA Container Toolkit

Reference: [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Install prerequisites:**

```bash
sudo apt-get update && sudo apt-get install -y curl gnupg2
```

**Configure the repository:**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

**Install the toolkit:**

```bash
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

### Configure Docker for GPU

**Configure the runtime:**

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

**Restart Docker:**

```bash
sudo systemctl restart docker
```

---

## Output

For each patient, the pipeline generates a PDF report containing:

- **Risk Score:** A quantitative measure derived from tissue morphology
- **Clinical Interpretation:** Whether the patient is predicted to benefit from docetaxel
- **Prognostic Estimates:** Expected outcomes based on the risk stratification
- **Treatment Considerations:** Context for clinical decision-making

---

## Run Modes

The pipeline supports four different run modes depending on your input structure:

### 1. Single Slide Mode

Process one slide as one patient.

```bash
docker run --gpus all \
  -v /path/to/slides:/data/input_slides:ro \
  -v /path/to/output:/data/output \
  madabhushilabapic/apic:latest \
  -i "/data/input_slides/slide.svs" \
  -o /data/output
```

### 2. Batch Mode (Multiple Patients, One Slide Each)

Process a folder of slides where **each slide is a different patient**.

```bash
docker run --gpus all \
  -v /path/to/slides_folder:/data/input_slides:ro \
  -v /path/to/output:/data/output \
  madabhushilabapic/apic:latest \
  -i "/data/input_slides/" \
  -o /data/output
```

**Input structure:**
```
/path/to/slides_folder/
  ├── patient1_slide.svs    → Patient 1
  ├── patient2_slide.svs    → Patient 2
  └── patient3_slide.svs    → Patient 3
```

### 3. Multi-Slide Mode (One Patient, Multiple Slides)

Process multiple slides from **a single patient**. Features are averaged across all slides to generate one patient-level prediction.

```bash
docker run --gpus all \
  -v /path/to/patient_folder:/data/input_slides:ro \
  -v /path/to/output:/data/output \
  madabhushilabapic/apic:latest \
  -i "/data/input_slides/" \
  -o /data/output \
  --multi-slide
```

**Input structure:**
```
/path/to/patient_folder/        → Patient PT001
  ├── biopsy_core_1.svs
  ├── biopsy_core_2.svs
  └── biopsy_core_3.svs
```

**Output:** One aggregated prediction and report for the patient (folder name = patient ID).

### 4. Batch Multi-Slide Mode (Multiple Patients, Multiple Slides Each)

Process multiple patients where **each patient has multiple slides**. Each subfolder is treated as a separate patient.

```bash
docker run --gpus all \
  -v /path/to/patients_folder:/data/input_slides:ro \
  -v /path/to/output:/data/output \
  madabhushilabapic/apic:latest \
  -i "/data/input_slides/" \
  -o /data/output \
  --multi-slide
```

**Input structure:**
```
/path/to/patients_folder/
  ├── PT001/                    → Patient PT001
  │   ├── biopsy_core_1.svs
  │   └── biopsy_core_2.svs
  ├── PT002/                    → Patient PT002
  │   ├── biopsy_core_1.svs
  │   ├── biopsy_core_2.svs
  │   └── biopsy_core_3.svs
  └── PT003/                    → Patient PT003
      └── biopsy_core_1.svs
```

**Output:** One aggregated prediction and report per patient.

### Mode Selection Summary

| Mode | Input | Flag | Description |
|------|-------|------|-------------|
| Single slide | File path | (none) | One slide = one patient |
| Batch | Folder with slides | (none) | Each slide = different patient |
| Multi-slide | Folder with slides | `--multi-slide` | All slides = one patient |
| Batch multi-slide | Folder with subfolders | `--multi-slide` | Each subfolder = one patient |

> **Auto-detection:** When using `--multi-slide`, the pipeline automatically detects whether the input folder contains slides directly (single patient) or subfolders (batch of patients).

### Additional Options

| Flag       | Description                                                              |
|------------|--------------------------------------------------------------------------|
| `--resume` | Skip already-processed slides. Useful for resuming interrupted batch jobs. |

**Example with resume:**

```bash
docker run --gpus all \
  -v /path/to/slides_folder:/data/input_slides:ro \
  -v /path/to/output:/data/output \
  madabhushilabapic/apic:latest \
  -i "/data/input_slides/" \
  -o /data/output \
  --resume
```

### Multi-Slide Feature Aggregation

When processing multiple slides per patient:

1. Each slide is processed independently through feature extraction
2. Features from all slides are **averaged using nanmean** (ignoring missing values)
3. The tissue overlay in the report comes from the slide with the **largest tissue area**
4. A single prediction and PDF report is generated per patient

---

## Citation

If you use this pipeline in your research, please cite:

> Medina S, Tokuyama N, Hammouda K, Pathak T, Mirtti T, Fu P, Gupta S, Lal P, Sandler HM, Correa R, Chafe S, Shah A, Efstathiou JA, Hoffman K, Straza M, Hallman MA, Jordan R, Pugh SL, Sweeney CJ, Madabhushi A. **A Computational Pathology Model to Predict Docetaxel Benefit in Localized High-Risk and Metastatic Prostate Cancer.** *Clin Cancer Res* 2025. [DOI: 10.1158/1078-0432.CCR-25-3327](https://doi.org/10.1158/1078-0432.CCR-25-3327)

---

## Disclaimer

**For research purposes only.** This tool is not intended for clinical use and should not be used to diagnose, treat, or make clinical decisions for any patient. The predictions generated by this pipeline have not been validated for clinical practice and are meant solely for academic and research applications.

## License

This software is licensed under the [Emory University License](LICENSE) for **non-commercial research purposes only**. See the LICENSE file for full terms and conditions.

For commercial licensing inquiries, contact Emory University.
