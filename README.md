---
title: "Data Analyst Agent"
thumbnail: "🧠"
emoji: "📊"
colorFrom: "indigo"
colorTo: "blue"
sdk: "docker"
app_file: "main.py"
license: "mit"
tags:
  - fastapi
  - data-analysis
  - python
  - agent
  - docker
  - huggingface
  - api
model: ""
space_holder: true
---



# 🧠 Data Analyst Agent - FastAPI

This is a **FastAPI-based Data Analyst Agent** that can perform **data analysis, web scraping, and visualization** through API calls.  
It integrates with **Anthropic Claude** for automated Python code generation and executes the code securely.

The app is designed for deployment on **[Hugging Face Spaces](https://huggingface.co/spaces)** using **Docker**.

---

## 🚀 Features
- **Web scraping** with `requests` + `BeautifulSoup`
- **Data analysis** with `pandas` & `duckdb`
- **Visualizations** with `matplotlib`, `seaborn`, and `plotly`
- **File uploads** for CSV, JSON, TXT datasets
- **Automated Python code generation via Claude**
- **CORS enabled** for web integration
- API response formats support JSON, images (Base64), or text

---

## 📂 Project Structure
├── main.py # FastAPI app entry point
├── claude_client.py # Claude API integration
├── code_executor.py # User Python code execution environment
├── requirements.txt # Python dependencies
├── Dockerfile # Hugging Face Docker deployment file
├── .env # Local environment secrets (ignored in git)
└── README.md # Documentation (this file)

---

## ⚡ Local Development

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Run the app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Then visit:
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🌐 Deploying to Hugging Face Spaces (Docker)

Hugging Face Spaces requires apps to listen on **port 7860** for Docker-based deployments.

### 1️⃣  `requirements.txt`

---

### 2️⃣ `Dockerfile`

**Important:** Port **7860** is required on Hugging Face Docker Spaces.

---

## 📡 API Endpoints

https://assathe-data-analyst-agent2.hf.space/
to check the root status.

https://assathe-data-analyst-agent2.hf.space/health
for the health endpoint.

https://assathe-data-analyst-agent2.hf.space/docs
for the Swagger docs and API playground 


### `POST /api/`
curl -X POST "https://assathe-data-analyst-agent2.hf.space/api/" \
  -F "questions=@questions.txt" \
  -F "files=@edges.csv" # (optional)

Send a question and optional files for analysis.
- **Form fields**:
  - `questions` → text file containing the question
  - `files` → one or more data files (CSV, JSON, TXT)
- **Returns**: JSON with results or Base64-encoded image

Example:
curl -X POST "https://assathe-data-analyst-agent2.hf.space/api/" \
-F "questions=@question.txt"
-F "files=@data.csv # (optional)

---

### `GET /health`
[Detailed health probe.](https://assathe-data-analyst-agent2.hf.space/health
for the health endpoint.)

---

## 🔑 Environment Variables
In your `.env` or Hugging Face Secrets panel:
- `CLAUDE_API_KEY` → Your Anthropic Claude API key
- Any other secrets needed for custom integrations

---

## 🛠 Troubleshooting
- **Build failure**: Check Docker build logs in Hugging Face Spaces.
- **Missing modules**: Ensure all dependencies are in `requirements.txt`.
- **Port issues**: Ensure Dockerfile runs on port **7860**.
- **Claude API errors**: Verify `CLAUDE_API_KEY` is set correctly in Hugging Face secrets.

---

## 🤝 Contributing
Pull requests are welcome. Please ensure:
- **Code is tested locally**
- **Dependencies are updated in `requirements.txt`**
- **Secrets are never committed to the repo**

---

## 📜 License
MIT License 

---

## 📌 Notes for Hugging Face Deployment
- Create a **new Space** and select **Docker** as SDK  
- Push the repo with `git push` to the provided Space repo link  
- Set **Secrets** in the Space settings (e.g., `CLAUDE_API_KEY`)  
- Hugging Face will automatically build and start the container




