# APIC Project

## Description
The **APIC Project** is an automated analysis pipeline that generates a **patient-level PDF report** from a **Whole Slide Image (WSI)**.  
For each patient, the pipeline performs a complete computational pathology analysis and outputs a **risk score with clinical interpretation** in an automatically generated PDF report.

---

## GPU Support (Optional)

This pipeline supports **GPU acceleration** using the **NVIDIA Container Toolkit**.  
If a GPU is **not available**, the pipeline will **automatically fall back to CPU execution**.

---

## NVIDIA Container Toolkit Installation  
(For Ubuntu / Debian-based systems)

> Reference:  
> https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

> **Note**  
> These instructions should work for any Debian-derived distribution.

---

### Install Prerequisites
#### With `apt`: Ubuntu, Debian
1. Install the prerequisites for the instructions below:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
  curl \
  gnupg2
```

2. Configure the production repository:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
Optionally, configure the repository to use experimental packages:

```bash
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
3. Update the packages list from the repository:

```bash
sudo apt-get update
```
4. Install the NVIDIA Container Toolkit packages:

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

### Configuration
#### Prerequisites
<li>
  You installed Docker.</li>
 <li>
  You installed the NVIDIA Container Toolkit.
</li>

### Configuring Docker
1. Configure the container runtime by using the nvidia-ctk command:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

The nvidia-ctk command modifies the /etc/docker/daemon.json file on the host. The file is updated so that Docker can use the NVIDIA Container Runtime.

2. Restart the Docker daemon:
```bash
sudo systemctl restart docker
```

## Running the APIC Pipeline
### Docker Pull command
```bash
sudo docker pull madabhushilabapic/apic:latest
```

### Docker Run Command (GPU Enabled)
```bash
sudo docker run --gpus all --rm \
  -v /path/to/root/input/slides:/data/input_slides:ro \
  -v /path/to/root/output:/data/output \
  madabhushilabapic/apic:latest \
  -i "/data/input_slides/slide_filename.ext" \
  -o /data/output
```

#### Explanation of Docker Paths
```bash
-v /path/to/root/input/slides:/data/input_slides:ro
```
<li><b>Host path</b>: Location of input Whole Slide Images</li>
<li><b>Container path</b>: `/data/input_slides`</li>
<li>`:ro` -> Mounted as read-only to prevent accidental modification</li>

```bash
-v /path/to/root/output:/data/output
```
<li><b>Host path</b>: Location of input Whole Slide Images</li>
<li><b>Container path</b>: `/data/input_slides`</li>
<li>Stores generated reports and intermediate results</li>

```bash
-i "/data/input_slides/slide_filename.ext"
```
<li> Input WSI file inside the container</li>

```bash
-o /data/putput
```
<li> Output directory inside the container</li>

### Output
For each patient, the pipeline generates a **comprehensive PDF report** that includes:
<li>Patient-specific risk score</li>
<li>Interpretation of the risk score</li>  
<li>Prognostic estimates </li> 
<li>Treatment considerations</li>

### Caveats
<li>Currently, the pipeline accepts one slide per patient</li>
<li>In batch mode, <b>each slide is assumed to belong to a different patient</b></li>

### Future Improvements
<li>Support multiple slides per patient</li>
<li>Accept a folder of slides per patient</li>
<li>Aggregate and average features across slides</li>
<li>Generate a <b>single patient-level summary risk score</b></li>

### Citation
If you use this pipeline in your research, please cite our work:

[Sebastian Medina, Naoto Tokuyama, Kamal Hammouda, Tilak Pathak, Tuomas Mirtti, Pingfu Fu, Shilpa Gupta, Priti Lal, Howard M. Sandler, Rohann Correa, Susan Chafe, Amit Shah, Jason A. Efstathiou, Karen Hoffman, Michael Straza, Mark A. Hallman, Richard Jordan, Stephanie L. Pugh, Christopher J. Sweeney, Anant Madabhushi; A Computational Pathology Model to Predict Docetaxel Benefit in Localized High-Risk and Metastatic Prostate Cancer. Clin Cancer Res 2025; https://doi.org/10.1158/1078-0432.CCR-25-3327]
 
