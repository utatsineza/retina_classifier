# Retina DR Classifier — ML Pipeline Summative
### African Leadership University | BSE | Machine Learning Pipeline

A complete end-to-end ML pipeline for **Diabetic Retinopathy Detection** using EfficientNetB0 transfer learning, deployed on AWS EC2.

---

## 🎥 Video Demo
[YouTube Demo Link](#) ← Add your YouTube link here after recording

## 🌐 Live URL
- **API:** http://13.61.0.203:8000/docs
- **UI Dashboard:** http://13.61.0.203:8080

---

## 📋 Project Description
This project classifies retinal fundus images into 5 grades of Diabetic Retinopathy (No DR, Mild, Moderate, Severe, Proliferative DR) using:
- **Model:** EfficientNetB0 (Transfer Learning from ImageNet)
- **Dataset:** APTOS 2019 Blindness Detection (3,662 images)
- **Backend:** FastAPI on AWS EC2
- **UI:** HTML/JS Dashboard with 4 tabs
- **Containerization:** Docker
- **Load Testing:** Locust

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/retina_classifier.git
cd retina_classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model
Download the trained model from Google Drive and place in `models/`:
- [efficientnet_retina.h5](https://drive.google.com/file/d/16SrroCKSCleqnW7JSjfWd8MpyInx3pKT/view?usp=sharing)
- [best_model.h5](https://drive.google.com/file/d/1RQB0Zg1a8RNFVEDeFH6gyXgGoLtXtmVm/view?usp=sharing)

### 4. Run with Docker
```bash
docker build -t retina-classifier .
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name retina-api \
  retina-classifier
```

### 5. Run UI
```bash
python3 -m http.server 8080 --directory ui
```

### 6. Run Locust Load Test
```bash
pip install locust
locust -f locust/locustfile.py --host=http://localhost:8000 --headless -u 10 -r 2 --run-time 60s
```

---

## 📊 Model Evaluation Results
| Metric | Value |
|--------|-------|
| Accuracy | 46.0% |
| F1 Score | 0.290 |
| Precision | 0.092 |
| Recall | 0.302 |
| ROC-AUC | 0.499 |

---

## 🔥 Load Testing Results (1 Docker Container)
| Users | Avg Latency | Median | 95th % | Req/s | Failures |
|-------|-------------|--------|--------|-------|----------|
| 10 | 728ms | 150ms | 360ms | 3.08 | 0% |
| 50 | 1501ms | 1500ms | 2300ms | 7.97 | 0% |
| 100 | 3427ms | 3500ms | 5200ms | 8.14 | 0% |

**Observation:** Latency increases significantly under heavy load with a single container on t3.micro. A load balancer with multiple containers would improve performance.

---

## 📁 Project Structure
```
retina_classifier/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── notebook/
│   └── retina_classifier.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── api/
│   └── main.py
├── ui/
│   └── index.html
├── locust/
│   └── locustfile.py
├── data/
│   ├── train/
│   └── test/
└── models/
    ├── best_model.h5
    └── efficientnet_retina.h5
```
