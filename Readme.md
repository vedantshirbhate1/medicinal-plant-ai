# 🌿 Indian Medicinal Plant AI Identifier

An AI-powered mobile application that identifies **Indian medicinal plants** from images and provides **detailed medicinal information**, powered by **Flask, PyTorch, React Native, and Google Gemini**.

---

## 🚀 Tech Stack

### **Frontend**
- React Native (Expo)
- TypeScript

### **Backend**
- Python (Flask)
- PyTorch (EfficientNetV2)

### **AI**
- Google Gemini API

---

## 📋 Prerequisites

Ensure the following are installed:

- Node.js  
- Python 3.8+  
- Git  
- Expo Go App (Android/iOS)  
- Phone + Laptop must be on same WiFi  

---

## ⚙️ Step 1 — Clone the Repository

```bash

git clone <YOUR_REPOSITORY_LINK>
cd Indian-Medicinal-Plant-AI

🐍 Step 2 — Backend Setup (Flask + PyTorch + Gemini)

Navigate to backend folder:

cd backend

1️⃣ Create Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate


macOS / Linux:

python3 -m venv venv
source venv/bin/activate

2️⃣ Install Python Dependencies
pip install -r requirements.txt

3️⃣ Configure Environment Variables

Create a .env file inside backend/:

GEMINI_API_KEY=your_actual_api_key_here

4️⃣ Ensure Model File Exists

Required model file:

backend/best_model.pth   (~160MB)


(If missing, obtain from the developer.)

5️⃣ Start Flask Server
python app.py


Expected output:

Running on http://0.0.0.0:5000

📱 Step 3 — Frontend Setup (React Native Expo)

Open a new terminal:

cd frontend

1️⃣ Install Node Modules
npm install

2️⃣ Set Backend IP Address

Find your IPv4 address:

Windows

ipconfig


Mac/Linux

ifconfig


Example:

192.168.1.45


Open this file:

frontend/app/(tabs)/index.tsx


Replace:

const SERVER_IP = "192.168.1.X";


With:

const SERVER_IP = "192.168.1.45";

3️⃣ Start Expo Development Server
npx expo start


Scan the QR code using Expo Go.

🛠 Troubleshooting
🔹 1. "Network request failed"

Use correct SERVER_IP

Ensure phone + laptop on same WiFi

Restart Flask & Expo

Allow Python through Windows Firewall

🔹 2. Missing Modules

Backend

pip install -r requirements.txt


Frontend

npm install

🔹 3. PowerShell Script Error (Windows)

Run as Administrator:

Set-ExecutionPolicy RemoteSigned

📂 Project Structure
Indian-Medicinal-Plant-AI/
│
├── backend/
│   ├── app.py
│   ├── best_model.pth
│   └── requirements.txt
│
└── frontend/
    ├── app/
    │   └── (tabs)/index.tsx
    ├── package.json
    └── app.json

🌱 You're Ready!

Backend identifies plant → Gemini generates botanical explanation → App displays detailed medicinal info.
