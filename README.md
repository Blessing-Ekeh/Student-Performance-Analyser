# 🎓 Student Performance Analyser

### A Multi-Agent Generative AI System powered by Azure OpenAI


https://github.com/user-attachments/assets/50cdabaa-917e-4093-a9b3-510e6ae4e2a8



## 📌 Project Overview

This project is a multi-agent AI system that analyses student academic performance data to uncover patterns, identify at-risk students, and generate actionable academic insights.

Built as part of a Generative AI module, the system demonstrates how specialised AI agents can collaborate in a structured pipeline to solve a real-world educational analytics problem.

The system processes student data, performs analysis through multiple agents, generates structured insights, and visualises key findings.

---

## 🏗️ Architecture

DataLoaderAgent → PerformanceAnalystAgent → RiskEvaluatorAgent → ReportGeneratorAgent

The system follows a **Sequential Multi-Agent Pipeline**, where each agent performs a specific task and passes its output to the next stage.

---

## 🤖 Agents

| Agent                   | Role                                                        | Tools Used                |
| ----------------------- | ----------------------------------------------------------- | ------------------------- |
| DataLoaderAgent         | Loads and summarises student performance data               | Data loading functions    |
| PerformanceAnalystAgent | Analyses grade distributions and performance trends         | Statistical analysis      |
| RiskEvaluatorAgent      | Identifies at-risk students based on performance thresholds | Custom logic / rules      |
| ReportGeneratorAgent    | Generates structured insights and summaries                 | None (aggregates outputs) |

All agents inherit from a shared base `Agent` class, which manages:

* LLM interaction (Azure OpenAI)
* Function/tool calling
* Iteration control (`max_iterations`)
* Error handling

---

## 🛠️ Tech Stack

* **LLM**: Azure OpenAI (GPT-4o)
* **Framework**: Custom multi-agent pipeline
* **Data Processing**: Pandas, NumPy
* **Visualisation**: Matplotlib, Seaborn
* **Interface**: Jupyter Notebook / Streamlit

---

## ✅ Key Features

* Multi-agent architecture for modular analysis
* Automated grade distribution analysis
* Pass vs Fail classification using threshold logic
* Risk detection for underperforming students
* Structured output generation
* Data visualisation for interpretability

---

## 📊 Output

The system produces:

### 1. Performance Analysis

* Distribution of final grades (G3)
* Identification of performance clusters

### 2. Risk Evaluation

* Detection of students below performance threshold
* Highlighting missing or insufficient risk factor data

### 3. Visualisations

* Grade distribution histogram
* Pass vs Fail segmentation

### 4. Structured Report

## ⚠️ Data Limitations

Some analyses returned *"No risk factor data"*, indicating:

* Missing or incomplete features in the dataset
* Need for improved feature engineering
* Real-world data quality challenges

---

## 📁 Project Structure

```
student-performance-analyser/
│
├── notebooks/
│   └── analysis.ipynb
├── app.py/
├── agents.py/
├── outputs/
├── README.md
├── requirements.txt
```

---

## ⚙️ Setup Instructions

1. Clone the repository:

```
git clone https://github.com/your-username/student-performance-analyser.git
```

2. Navigate into the project:

```
cd student-performance-analyser
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the notebook:

```
jupyter notebook
```

---

## 🔑 Azure OpenAI Setup

To run this project, you need:

* Azure OpenAI API key
* Endpoint
* Deployment name and version

Store them in a `.env` file:

```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```
---

## 👤 Author

**Blessing Ekeh**

Data Science & Generative AI Enthusiast

---
