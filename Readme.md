# 🌿 Indian Medicinal Plant AI Identifier

An AI-powered mobile application that identifies **Indian medicinal plants** from images, provides **detailed medicinal properties**, and includes an **AI botanical expert chatbot** powered by Google Gemini.

---

## 🚀 Tech Stack

### **Frontend**
- React Native (Expo)
- TypeScript

### **Backend**
- Python (Flask)
- PyTorch (EfficientNetV2 Model)

### **AI Services**
- Google Gemini API (Medicinal Plant Knowledge)

---

## 📋 Prerequisites

Install the following before running the project:

- **Node.js**
- **Python 3.8+**
- **Expo Go App** (Android/iOS)
- **Git**
- Ensure **phone + laptop are on the same WiFi**

---

## ⚙️ Step 1: Clone the Project


git clone <YOUR_REPOSITORY_LINK>
cd Indian-Medicinal-Plant-AI

🐍 Step 2: Backend Setup (Flask + PyTorch + Gemini)
Navigate to backend folder:
cd backend

1️⃣ Create Virtual Environment

Windows
python -m venv venv
.\venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Environment Variables

Create .env file in backend/:
GEMINI_API_KEY=your_actual_api_key_here

4️⃣ Model File Check
Ensure the model exists:

backend/best_model.pth   (≈160MB)
If missing, obtain the file from the developer.

5️⃣ Start Flask Server

python app.py

Expected output:

Running on http://0.0.0.0:5000 //e.g.

📱 Step 3: Frontend Setup (React Native)
Open new terminal:
cd frontend

1️⃣ Install Node Modules

npm install

2️⃣ Configure Backend IP (IMPORTANT)
Find your computer’s local IP:

Windows:
ipconfig

Mac/Linux:
ifconfig

Locate IPv4 Address, e.g.:
Copy code
192.168.1.45

Now open:
frontend/app/(tabs)/index.tsx

Replace the existing line:

const SERVER_IP = "192.168.1.X";
with your actual IP.

3️⃣ Start Expo App

npx expo start
Scan the QR code using Expo Go on your mobile.

🛠 Troubleshooting

1. Network Request Failed

Phone + laptop must be on same WiFi
Allow Python through Windows Firewall
Ensure correct IP in index.tsx
Restart both backend and Expo

2. Missing Modules

Backend:

pip install -r requirements.txt

Frontend:

npm install

3. PowerShell Script Execution Error (Windows)
Run this as Administrator:

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

🌱 You're Ready to Run the App!
Backend will classify medicinal plants → send result → Gemini generates botanical explanation → App displays detailed info.