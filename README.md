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
  --multi-slide \
  --patient-id Patient001
```

> **Note:** `--patient-id` is required for single-patient multi-slide mode. This sets the patient identifier used in output files and the report.

**Input structure:**

```text
/path/to/patient_folder/        → Mounted as /data/input_slides
  ├── biopsy_core_1.svs
  ├── biopsy_core_2.svs
  └── biopsy_core_3.svs
```

**Output:** `Patient001_report.pdf` with aggregated prediction.

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

### 5. Research Mode (Batch with Auto-Grouping)

Process a flat folder of slides where **patient ID is extracted from the filename**. Slides are automatically grouped by patient ID, features are averaged per patient, and a summary CSV is generated. **No PDF reports are generated** in this mode.

```bash
docker run --gpus all \
  -v /path/to/slides_folder:/data/input_slides:ro \
  -v /path/to/output:/data/output \
  madabhushilabapic/apic:latest \
  -i "/data/input_slides/" \
  -o /data/output \
  --research-mode
```

**Patient ID Extraction:**

The patient ID is extracted as the **number before the first underscore** in the filename:

```text
<patient_id>_<rest_of_filename>.svs
     ↑
 Number before first underscore = patient ID
```

**Examples:**

```text
001_biopsy_core_A.svs     → patient_id = "001"
042_sample_tissue.svs     → patient_id = "042"
042_another_core.svs      → patient_id = "042" (grouped with above)
```

**Input structure:**
```
/path/to/slides_folder/
  ├── 001_biopsy_core_A.svs       → Patient 001 (1 slide)
  ├── 042_sample_tissue.svs       → Patient 042 (2 slides)
  ├── 042_another_core.svs        → Patient 042 (grouped)
  └── 103_tissue_section.svs      → Patient 103 (1 slide)
```

**Output:**
```
/path/to/output/
  ├── research_results.csv        ← Summary CSV with all patients
  ├── 001/                         ← Per-patient aggregated features
  │   ├── final_features/
  │   └── qc/
  ├── 042/
  │   ├── final_features/
  │   └── qc/
  ├── 103/
  │   ├── final_features/
  │   └── qc/
  └── [individual slide outputs]   ← Preserved for each slide
```


> **Note:** Slides that don't match the expected naming pattern (number before first underscore) will be skipped with a warning.

### Mode Selection Summary

| Mode              | Input                  | Flags                         | Description                      |
| ----------------- | ---------------------- | ----------------------------- | -------------------------------- |
| Single slide      | File path              | (none)                        | One slide = one patient          |
| Batch             | Folder with slides     | (none)                        | Each slide = different patient   |
| Multi-slide       | Folder with slides     | `--multi-slide --patient-id`  | All slides = one patient         |
| Batch multi-slide | Folder with subfolders | `--multi-slide`               | Each subfolder = one patient     |
| Research          | Folder with slides     | `--research-mode`             | Auto-group by filename, CSV out  |

> **Auto-detection:** When using `--multi-slide`, the pipeline automatically detects whether the input folder contains slides directly (single patient, requires `--patient-id`) or subfolders (batch of patients, uses subfolder names as patient IDs).

### Additional Options

| Flag              | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `--resume`        | Skip already-processed slides. Useful for resuming interrupted batch jobs.  |
| `--patient-id`    | Patient identifier for single-patient multi-slide mode (required).          |
| `--research-mode` | Enable research mode: auto-group slides by patient ID, output CSV summary.  |

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
3. The tissue overlay in the report comes from the **first available slide**
4. A single prediction and PDF report is generated per patient

### Patient ID and Report Naming

The patient ID (and report filename) is determined as follows:

| Mode              | Patient ID Source       | Report Name                     |
|-------------------|-------------------------|---------------------------------|
| Single slide      | Slide filename (stem)   | `{slide_name}_report.pdf`       |
| Batch             | Each slide filename     | `{slide_name}_report.pdf`       |
| Multi-slide       | `--patient-id` flag     | `{patient_id}_report.pdf`       |
| Batch multi-slide | Each subfolder name     | `{subfolder_name}_report.pdf`   |

**Example for multi-slide (single patient):**

```text
# Your local folder:
/home/user/PatientA/
  ├── biopsy_1.svs
  └── biopsy_2.svs

# Docker command:
docker run ... -v /home/user/PatientA:/data/input_slides:ro \
  -i /data/input_slides/ --multi-slide --patient-id PatientA

# Output: PatientA_report.pdf
```

**Example for batch multi-slide (multiple patients):**

```text
# Your local folder structure:
/home/user/patients/
  ├── PatientA/
  │   ├── biopsy_1.svs
  │   └── biopsy_2.svs
  └── PatientB/
      └── biopsy_1.svs

# Docker command:
docker run ... -v /home/user/patients:/data/input_slides:ro \
  -i /data/input_slides/ --multi-slide

# Output: PatientA_report.pdf, PatientB_report.pdf
```

> **Important:** Name your patient folders with the desired patient ID. The folder name becomes the patient identifier in all outputs.

---

## Contributing

### Development Workflow

1. **Create a branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```
   All 98 tests must pass before pushing.

4. **Commit**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push and create PR**
   ```bash
   git push -u origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

### Test Suite

The test suite validates pipeline behavior without requiring model inference:

| Test File | Coverage |
|-----------|----------|
| `test_modes.py` | Mode detection (single, batch, multi-slide, resume) |
| `test_patient_pipeline.py` | Multi-slide aggregation |
| `test_step_interfaces.py` | Step input/output contracts |
| `test_pipeline_integration.py` | End-to-end data flow |
| `test_symlink_handling.py` | Resume after interruption |
| `test_feature_aggregation.py` | Feature averaging with NaN handling |

Tests run in ~10 seconds using dummy data. They are excluded from the Docker image via `.dockerignore`.

### Building and Pushing to Docker Hub

To build and push a new version to Docker Hub:

```bash
./build_and_push.sh
```

This script will:

1. Prompt for a version number (e.g., `1.0.7`)
2. Build the image as `madabhushilabapic/apic:v1.0.7`
3. Tag it as `madabhushilabapic/apic:latest`
4. Push both tags to Docker Hub

> **Note:** Requires Docker Hub login (`docker login`) and push access to the repository.

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
