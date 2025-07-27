# DevOps AI Flask API 🧠🚀

This project serves a machine learning model using Flask, deployed via Kubernetes with Helm, containerized with Docker, and integrated into a CI/CD pipeline using GitHub Actions.

## Features
- 📦 Flask-based REST API
- 🔍 Dummy ML model served with `/predict` endpoint
- ❤️ Health check via `/health`
- 🐳 Dockerized build with multi-stage optimization
- ⛵ Helm chart for Kubernetes deployment
- 🔁 Rollback support via Helm + GitHub Actions

## API Usage
POST `/predict`

```json
{
  "features": [0, 0]
}