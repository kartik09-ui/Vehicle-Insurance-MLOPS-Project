
---

# 🚗 Vehicle Insurance Response Prediction — End-to-End MLOps Pipeline

An **end-to-end Machine Learning Operations (MLOps) project** designed to predict customer interest in vehicle insurance. This repository demonstrates a production-grade MLOps architecture featuring modular Python packaging, MongoDB data ingestion, AWS S3 model registry management, FastAPI REST service delivery, and automated CI/CD using Docker, AWS ECR, and a self-hosted AWS EC2 GitHub Actions runner.

---

## 🌟 Key Project Highlights

* **Modular Python Architecture:** Organized via `setup.py` and `pyproject.toml` for local library imports and scalability.
* **Dynamic Data Ingestion:** Reads raw data dynamically from **MongoDB Atlas** into pandas DataFrames.
* **AWS S3 Model Registry:** Automatic evaluation comparing candidate models against production models stored on AWS S3 using performance thresholding.
* **Asynchronous Web Service:** Low-latency **FastAPI** web application with real-time model inference and an interactive HTML UI.
* **On-Demand Retraining:** Dedicated `/train` endpoint allowing dynamic triggering of the training pipeline.
* **Automated CI/CD Pipeline:** Fully automated workflow built with **GitHub Actions**, **Docker Containerization**, **AWS ECR**, and a **Self-Hosted AWS EC2 Runner**.

---

## 🛠️ Tech Stack & MLOps Tooling

| Domain | Technologies / Services Used |
| --- | --- |
| **Language & Environment** | Python 3.10 / 3.12, Conda |
| **Data Management** | MongoDB Atlas, Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Cloud Infrastructure (AWS)** | AWS S3, AWS ECR, AWS EC2 (Ubuntu 24.04), AWS IAM |
| **Web Framework** | FastAPI, Uvicorn, Jinja2, HTML5/CSS3 |
| **Containerization & Deployment** | Docker, GitHub Actions (Self-Hosted Runner) |

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│  MongoDB Atlas  │ ────> │ Data Ingestion  │ ────> │ Data Validation  │
└─────────────────┘       └─────────────────┘       └────────┬─────────┘
                                                             │
┌─────────────────┐       ┌─────────────────┐                ▼
│ Model Evaluation│ <──── │  Model Trainer  │ <──── ┌──────────────────┐
│    & Pusher     │       └─────────────────┘       │Data Transformation│
└────────┬────────┘                                 └──────────────────┘
         │
         ▼
 ┌──────────────┐          ┌─────────────────────────────────────────────────┐
 │    AWS S3    │ ───────> │            GitHub Actions CI/CD                 │
 │ Model Bucket │          │  (Build Image ➔ ECR Push ➔ EC2 Deploy & Run)   │
 └──────────────┘          └────────────────────────┬────────────────────────┘
                                                    │
                                                    ▼
                                           [ FastAPI App @ Port 5000 ]

```

---

## 📌 Project Execution & Setup Walkthrough

### Step 1–4: Environment & Local Packaging Setup

1. Execute `template.py` to construct project folder structure.
2. Configure local library imports in `setup.py` and `pyproject.toml`.
3. Create and activate virtual environment:
```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt

```


4. Confirm local package installation:
```bash
pip list

```



### Step 5–13: Database Connection (MongoDB Atlas)

1. Provision a MongoDB Atlas cluster (M0 free tier).
2. Set Network Access to `0.0.0.0/0` and generate database credentials.
3. Retrieve Python driver connection string.
4. Push dataset to MongoDB via `notebook/MongoDB_demo.ipynb`.
5. Verify collection creation on MongoDB Atlas web console.

### Step 14–16: Core Utilities & Exploratory Data Analysis

1. Write custom `logger.py` and `exception.py` utilities and test via `demo.py`.
2. Add EDA and Feature Engineering Jupyter notebooks.

### Step 17–18: Data Ingestion & Environment Configuration

1. Define constants in `src/constants/__init__.py`.
2. Configure MongoDB connection helper (`src/configuration/mongo_db_connection.py`).
3. Build `DataIngestionConfig`, `DataIngestionArtifact`, and `DataIngestionComponent` pipelines.
4. Export environment connection variables:
* **Linux/macOS Bash:** `export MONGODB_URL="mongodb+srv://..."`
* **Windows PowerShell:** `$env:MONGODB_URL="mongodb+srv://..."`



### Step 19–23: Validation, Transformation, Training & AWS Setup

1. Define schema specs in `config/schema.yaml` and utility helpers in `src/utils/main_utils.py`.
2. Implement **Data Validation**, **Data Transformation**, and **Model Trainer** components.
3. Configure AWS IAM User (`firstproj`) with `AdministratorAccess` and store credentials:
```bash
export AWS_ACCESS_KEY_ID="<your-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"

```


4. Create an S3 Bucket (`my-model-mlopsproj`) in region `us-east-1` for model registry storage.
5. Build AWS S3 access modules (`src/aws_storage`) and S3 Estimator handlers (`src/entity/s3_estimator.py`).

### Step 24–26: Model Evaluation, Pusher & Application Layer

1. Build `ModelEvaluation` and `ModelPusher` pipeline components.
2. Implement FastAPI server in `app.py` and prediction pipeline classes.
3. Add `static/` and `templates/` folders for Jinja2 template rendering.

### Step 27–34: CI/CD Deployment with AWS ECR & Self-Hosted Runner

1. Create `Dockerfile` and `.dockerignore`.
2. Provision an AWS ECR repository named `vehicleproj`.
3. Launch an AWS EC2 Ubuntu 24.04 instance (`t2.medium`, 30 GB storage).
4. Install Docker on the EC2 instance:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu && newgrp docker

```


5. Register EC2 as a **GitHub Self-Hosted Runner** via GitHub Settings > Actions > Runners.
6. Set GitHub Secrets:
* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_DEFAULT_REGION`
* `ECR_REPO`
* `MONGODB_URL`


7. Update EC2 Security Group inbound rules to allow Custom TCP traffic on port `5000` (or `5080`).
8. Push code to trigger CI/CD pipeline and view live web application via `http://<EC2-PUBLIC-IP>:5000`.

---

## 📊 Model Input Feature Schema

| Parameter | Description | Data Type |
| --- | --- | --- |
| `Gender` | Gender (`1`: Male, `0`: Female) | Binary (`0`/`1`) |
| `Age` | Customer Age | Integer |
| `Driving_License` | License possession status (`1`: Yes, `0`: No) | Binary (`0`/`1`) |
| `Region_Code` | Customer region code | Float |
| `Previously_Insured` | Existing vehicle insurance (`1`: Yes, `0`: No) | Binary (`0`/`1`) |
| `Annual_Premium` | Annual premium payment amount | Float |
| `Policy_Sales_Channel` | Channel code used for outreach | Float |
| `Vintage` | Company association duration (days) | Integer |
| `Vehicle_Age_lt_1_Year` | Vehicle age under 1 year (`1`: Yes, `0`: No) | Binary (`0`/`1`) |
| `Vehicle_Age_gt_2_Years` | Vehicle age over 2 years (`1`: Yes, `0`: No) | Binary (`0`/`1`) |
| `Vehicle_Damage_Yes` | Previous vehicle damage (`1`: Yes, `0`: No) | Binary (`0`/`1`) |

---

## 🔌 API Reference

| Endpoint | Method | Functionality |
| --- | --- | --- |
| `/` | `GET` | Renders the HTML input prediction interface |
| `/` | `POST` | Processes web form inputs and returns predicted insurance response |
| `/train` | `GET` | Initiates the automated end-to-end ML model retraining pipeline |

---

## 💻 Running the App Locally

```bash
# Clone Repository
git clone https://github.com/your-username/Vehicle-Insurance-MLOPS-Project.git
cd Vehicle-Insurance-MLOPS-Project

# Activate Virtual Environment
conda activate vehicle

# Set Database URL
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/..."

# Start FastAPI Application
python app.py

```