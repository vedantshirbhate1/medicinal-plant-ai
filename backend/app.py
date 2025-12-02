import os
import io
import json
import base64
import requests
import numpy as np
import cv2 # Required for Albumentations transforms
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
from dotenv import load_dotenv

# --- PYTORCH/ML IMPORTS ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# Note: efficientnet_pytorch is used for loading EfficientNet base structure.
from efficientnet_pytorch import EfficientNet 
from rembg import remove # Used for background removal
from timm import create_model

# --- CONFIGURATION ---

# 1. Load Environment Variables
load_dotenv() # <--- ADD THIS: It loads variables from the .env file

# 2. Securely get the API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # <--- MODIFIED

if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")
    # You might want to exit here or handle it gracefully

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# 2. Model Path and Configuration
MODEL_PATH = 'best_model.pth'
IMG_SIZE = 384 
MODEL_NAME = 'tf_efficientnetv2_b3'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Class Mapping (Extracted from your Classification Report - COMPLETE LIST)
CLASS_NAMES = [
    "Allium Cepa (Onion)", "Aloe Vera (Aloevera)", "Alpinia Galanga (Rasna)", 
    "Amaranthus Viridis (Arive-Dantu)", "Andrographis Paniculata (Nelavembu)", 
    "Annona Squamosa (Custard Apple)", "Artocarpus Heterophyllus (Jackfruit)", 
    "Azadirachta Indica (Neem)", "Bambusa Vulgaris (Bamboo)", "Basella Alba (Basale)", 
    "Brassica Juncea (Indian Mustard)", "Brassica Oleracea Gongylodes (Kohlrabi)", 
    "Butea Monosperma (Palash)", "Calotropis Gigantea (Crown Flower)", 
    "Capsicum Annuum (Chilli)", "Cardiospermum Halicacabum (Balloon Vine)", 
    "Carica Papaya (Papaya)", "Carissa Carandas (Karanda)", 
    "Catharanthus Roseus (Madagascar Periwinkle)", "Centella Asiatica (Brahmi)", 
    "Chakte", "Chamaecostus Cuspidatus (Insulin Plant)", 
    "Cinnamomum Camphora (Camphor)", "Citrus Limon (Lemon)", "Coffea Arabica (Coffee)", 
    "Colocasia Esculenta (Taro)", "Common rue (naagdalli)", "Coriandrum Sativum (Coriander)", 
    "Cucurbita Pepo (Pumpkin)", "Cymbopogon (Lemongrass)", "Cyperus (Ganigale)", 
    "Eclipta Prostrata (Bhringraj)", "Eucalyptus Globulus (Eucalyptus)", 
    "Euphorbia Hirta (Asthma Weed)", "Ficus Auriculata (Roxburgh fig)", 
    "Ficus Religiosa (Peepal Tree)", "Gomphrena globosa (Globe Amarnath)", 
    "Graptophyllum pictum (Caricature)", "Hibiscus Rosa-sinensis", "Jasminum (Jasmine)", 
    "Justicia Adhatoda (Malabar Nut)", "Kambajala", "Kasambruga", "Lantana Camara (Lantana)",
    "Lawsonia Inermis (Henna)", "Leucas Aspera (Thumbe)", "Mangifera Indica (Mango)", 
    "Manilkara Zapota (Sapota)", "Mentha (Mint)", "Michelia Champaca (Sampige)", 
    "Morinda Citrifolia (Noni)", "Moringa Oleifera (Drumstick)", 
    "Muntingia Calabura (Jamaica Cherry-Gasagase)", "Murraya Koenigii (Curry)", 
    "Nerium Oleander (Oleander)", "Nyctanthes Arbor-tristis (Parijata)", 
    "Ocimum Tenuiflorum (Tulsi)", "Ocimum basilicum (kamakasturi)", 
    "Papaver Somniferum (Poppy)", "Phaseolus Vulgaris (Beans)", "Phyllanthus Emblica (Amla)", 
    "Piper Betle (Betel)", "Piper Nigrum (Pepper)", "Pisum Sativum (Pea)", 
    "Plectranthus Amboinicus (Mexican Mint)", "Pongamia Pinnata (Indian Beech)", 
    "Psidium Guajava (Guava)", "Punica Granatum (Pomegranate)", "Raphanus Sativus (Radish)", 
    "Ricinus Communis (Castor)", "Rosa (Rose)", "Santalum Album (Sandalwood)", 
    "Saraca Asoca (Ashoka)", "Saraca Asoca (Seethaashoka)", "Solanum Lycopersicum (Tomato)", 
    "Solanum Nigrum (Black Nightshade)", "Spinacia Oleracea (Spinach)", 
    "Stereospermum Chelonoides (Padri)", "Syzygium Cumini (Jamun)", 
    "Syzygium Jambos (Rose Apple)", "Tabernaemontana Divaricata (Crape Jasmine)", 
    "Tagetes (Marigold)", "Tamarindus Indica (Tamarind)", "Tecoma Stans (Yellow Bells)", 
    "Tinospora Cordifolia (Amruthaballi)", "Trigonella Foenum-graecum (Fenugreek)", 
    "Turmeric", "Zingiber Officinale (Ginger)", "kepala" 
]
NUM_CLASSES = len(CLASS_NAMES)

# 4. Simulated Database for 89 Classes (Uses must be verified and expanded by the user)
DB_PLANT_DATA = {
    "Allium Cepa (Onion)": {"is_medicinal": True, "uses": "Used in traditional medicine for its antiseptic properties, cold, and respiratory support."},
    "Aloe Vera (Aloevera)": {"is_medicinal": True, "uses": "Known for soothing skin burns, reducing inflammation, and aiding digestion (when prepared correctly)."},
    "Alpinia Galanga (Rasna)": {"is_medicinal": True, "uses": "Used for anti-inflammatory purposes, treating arthritis, and promoting respiratory health."},
    "Amaranthus Viridis (Arive-Dantu)": {"is_medicinal": True, "uses": "Traditionally used as a diuretic and for treating inflammation and digestive issues."},
    "Andrographis Paniculata (Nelavembu)": {"is_medicinal": True, "uses": "Potent immune booster, used to treat fever, flu, and upper respiratory tract infections."},
    "Annona Squamosa (Custard Apple)": {"is_medicinal": True, "uses": "Leaves are used to treat lice and roots are used for dysentery in some folk medicine practices."},
    "Artocarpus Heterophyllus (Jackfruit)": {"is_medicinal": True, "uses": "Roots are sometimes used for diarrhea and the ripe fruit is a mild laxative."},
    "Azadirachta Indica (Neem)": {"is_medicinal": True, "uses": "Strong antibacterial, antifungal, and blood purification properties. Widely used for skin diseases."},
    "Bambusa Vulgaris (Bamboo)": {"is_medicinal": True, "uses": "Young shoots are consumed for their high fiber and nutrient content; used in cough and lung remedies."},
    "Basella Alba (Basale)": {"is_medicinal": True, "uses": "A cooling herb, often used to soothe internal ulcers and treat constipation."},
    "Brassica Juncea (Indian Mustard)": {"is_medicinal": True, "uses": "Seeds are used externally as a rubefacient and internally for stimulating appetite."},
    "Brassica Oleracea Gongylodes (Kohlrabi)": {"is_medicinal": True, "uses": "Rich in antioxidants and traditionally consumed for anti-inflammatory benefits."},
    "Butea Monosperma (Palash)": {"is_medicinal": True, "uses": "Flowers are used for skin diseases and seeds are used as an anthelmintic (to expel worms)."},
    "Calotropis Gigantea (Crown Flower)": {"is_medicinal": True, "uses": "Used externally for treating swellings, joint pain, and skin irritations (Note: Highly toxic if ingested)."},
    "Capsicum Annuum (Chilli)": {"is_medicinal": True, "uses": "Contains capsaicin, used topically for pain relief in arthritis and internally to aid digestion."},
    "Cardiospermum Halicacabum (Balloon Vine)": {"is_medicinal": True, "uses": "Leaves and roots are used in Ayurvedic medicine to treat joint pain, rheumatism, and earaches."},
    "Carica Papaya (Papaya)": {"is_medicinal": True, "uses": "Contains papain, used to aid protein digestion and treat wounds. Leaves are studied for anti-dengue properties."},
    "Carissa Carandas (Karanda)": {"is_medicinal": True, "uses": "Ripe fruit is used to treat indigestion and its roots are used for temporary relief from pain."},
    "Catharanthus Roseus (Madagascar Periwinkle)": {"is_medicinal": True, "uses": "Source of vinca alkaloids (vinblastine, vincristine), used in chemotherapy (Note: Highly toxic)."},
    "Centella Asiatica (Brahmi)": {"is_medicinal": True, "uses": "Known as a nerve tonic, used to improve memory, cognitive function, and reduce anxiety."},
    "Chakte": {"is_medicinal": True, "uses": "Traditional use for minor ailments and nutritional supplementation (Placeholder: Please verify and expand)."},
    "Chamaecostus Cuspidatus (Insulin Plant)": {"is_medicinal": True, "uses": "Leaves are traditionally chewed to help manage blood sugar levels (needs medical oversight)."},
    "Cinnamomum Camphora (Camphor)": {"is_medicinal": True, "uses": "Used externally as a counterirritant for muscle pain; vapor is inhaled for respiratory congestion."},
    "Citrus Limon (Lemon)": {"is_medicinal": True, "uses": "Rich in Vitamin C, used for boosting immunity, and its essential oil has antiseptic qualities."},
    "Coffea Arabica (Coffee)": {"is_medicinal": True, "uses": "Source of caffeine, a stimulant used to combat fatigue and improve alertness."},
    "Colocasia Esculenta (Taro)": {"is_medicinal": True, "uses": "The corm and leaves are nutrient-rich and used in some preparations for skin conditions."},
    "Common rue (naagdalli)": {"is_medicinal": True, "uses": "Used traditionally for nervous disorders and external fungal infections (Note: Can be toxic)."},
    "Coriandrum Sativum (Coriander)": {"is_medicinal": True, "uses": "Used for digestive aid, reducing bloating, and its seeds have been studied for lowering cholesterol."},
    "Cucurbita Pepo (Pumpkin)": {"is_medicinal": True, "uses": "Seeds are used traditionally as a remedy for intestinal worms (anthelmintic)."},
    "Cymbopogon (Lemongrass)": {"is_medicinal": True, "uses": "Used as a febrifuge (to lower fever), anti-inflammatory agent, and for digestive comfort."},
    "Cyperus (Ganigale)": {"is_medicinal": True, "uses": "Aromatic roots are used in perfumes and traditionally for digestive ailments and as a diuretic."},
    "Eclipta Prostrata (Bhringraj)": {"is_medicinal": True, "uses": "Mainly used as a tonic for hair growth, liver disorders, and to improve overall vitality."},
    "Eucalyptus Globulus (Eucalyptus)": {"is_medicinal": True, "uses": "Essential oil is used for respiratory problems, easing congestion, and as an antiseptic."},
    "Euphorbia Hirta (Asthma Weed)": {"is_medicinal": True, "uses": "Used in folk medicine for treating asthma, cough, and other respiratory disorders."},
    "Ficus Auriculata (Roxburgh fig)": {"is_medicinal": True, "uses": "The fruit and bark are used in some preparations to treat diarrhea and dysentery."},
    "Ficus Religiosa (Peepal Tree)": {"is_medicinal": True, "uses": "Bark and fruits are used traditionally for treating asthma, constipation, and wound healing."},
    "Gomphrena globosa (Globe Amarnath)": {"is_medicinal": True, "uses": "Flowers are traditionally used to treat cough, bronchitis, and dysentery."},
    "Graptophyllum pictum (Caricature)": {"is_medicinal": True, "uses": "Leaves are used traditionally for treating constipation and external skin swelling."},
    "Hibiscus Rosa-sinensis": {"is_medicinal": True, "uses": "Flowers are used for hair conditioning and growth; roots and leaves are used for cough."},
    "Jasminum (Jasmine)": {"is_medicinal": True, "uses": "Flowers and oil are used for their aromatic, relaxing, and antiseptic properties in skincare."},
    "Justicia Adhatoda (Malabar Nut)": {"is_medicinal": True, "uses": "Potent cough suppressant and expectorant, widely used for bronchitis and asthma."},
    "Kambajala": {"is_medicinal": True, "uses": "Traditional use as a coolant and digestive aid (Placeholder: Please verify and expand)."},
    "Kasambruga": {"is_medicinal": True, "uses": "Used in folk remedies for general wellness and vitality (Placeholder: Please verify and expand)."},
    "Lantana Camara (Lantana)": {"is_medicinal": True, "uses": "Leaves are used externally to treat swellings and cuts (Note: Berries are toxic)."},
    "Lawsonia Inermis (Henna)": {"is_medicinal": True, "uses": "Used as a natural dye, and traditionally applied to the skin and hair for its cooling properties."},
    "Leucas Aspera (Thumbe)": {"is_medicinal": True, "uses": "Used for cough, cold, fever, and skin infections in various parts of India."},
    "Mangifera Indica (Mango)": {"is_medicinal": True, "uses": "Raw mango is used for treating heatstroke; bark is used for dysentery."},
    "Manilkara Zapota (Sapota)": {"is_medicinal": True, "uses": "Bark is used to treat diarrhea, and the seed is considered diuretic and useful in bladder stones."},
    "Mentha (Mint)": {"is_medicinal": True, "uses": "Used to relieve indigestion, irritable bowel syndrome, and tension headaches."},
    "Michelia Champaca (Sampige)": {"is_medicinal": True, "uses": "Flowers are used for their aromatic oil and in remedies for fever and vomiting."},
    "Morinda Citrifolia (Noni)": {"is_medicinal": True, "uses": "Fruit and leaves are used for general well-being, pain relief, and anti-inflammatory effects."},
    "Moringa Oleifera (Drumstick)": {"is_medicinal": True, "uses": "Highly nutritious; leaves are used to combat malnutrition and for anti-inflammatory properties."},
    "Muntingia Calabura (Jamaica Cherry-Gasagase)": {"is_medicinal": True, "uses": "Leaves are used to brew tea which has properties to reduce blood pressure and inflammation."},
    "Murraya Koenigii (Curry)": {"is_medicinal": True, "uses": "Used for digestive issues, managing morning sickness, and its antioxidant properties."},
    "Nerium Oleander (Oleander)": {"is_medicinal": True, "uses": "Used in some traditional topical applications for skin issues (WARNING: Extremely toxic if ingested)."},
    "Nyctanthes Arbor-tristis (Parijata)": {"is_medicinal": True, "uses": "Leaves are used to treat fevers, rheumatism, and sciatica."},
    "Ocimum Tenuiflorum (Tulsi)": {"is_medicinal": True, "uses": "Revered as an adaptogen, used for stress reduction, respiratory health, and immune support."},
    "Ocimum basilicum (kamakasturi)": {"is_medicinal": True, "uses": "Used for culinary and medicinal purposes; helps in digestion and relieves bloating."},
    "Papaver Somniferum (Poppy)": {"is_medicinal": True, "uses": "Seeds are used for their calming effect and as a source of healthy fats (source of opium, requires care)."},
    "Phaseolus Vulgaris (Beans)": {"is_medicinal": True, "uses": "Pods are used as a diuretic; rich source of fiber and protein for dietary health."},
    "Phyllanthus Emblica (Amla)": {"is_medicinal": True, "uses": "Rich in Vitamin C; used as a potent antioxidant, immune booster, and hair/skin conditioner."},
    "Piper Betle (Betel)": {"is_medicinal": True, "uses": "Used for its stimulating, antiseptic, and digestive properties; chewed with areca nut."},
    "Piper Nigrum (Pepper)": {"is_medicinal": True, "uses": "Used to enhance nutrient absorption (bioavailability enhancer) and aid digestion."},
    "Pisum Sativum (Pea)": {"is_medicinal": True, "uses": "High in protein and fiber, contributes to heart and digestive health."},
    "Plectranthus Amboinicus (Mexican Mint)": {"is_medicinal": True, "uses": "Leaves are used to treat coughs, sore throats, and cold symptoms."},
    "Pongamia Pinnata (Indian Beech)": {"is_medicinal": True, "uses": "Oil from the seeds is used externally to treat skin diseases, ulcers, and rheumatism."},
    "Psidium Guajava (Guava)": {"is_medicinal": True, "uses": "Leaves are brewed as a tea to treat diarrhea and have antiseptic properties for wound cleaning."},
    "Punica Granatum (Pomegranate)": {"is_medicinal": True, "uses": "Bark and fruit peel are used to treat diarrhea and dysentery; fruit is rich in antioxidants."},
    "Raphanus Sativus (Radish)": {"is_medicinal": True, "uses": "Used as a detoxifier and diuretic, traditionally believed to cleanse the liver and kidneys."},
    "Ricinus Communis (Castor)": {"is_medicinal": True, "uses": "Oil is a powerful laxative and is used externally for inflammation and joint pain (for external use only)."},
    "Rosa (Rose)": {"is_medicinal": True, "uses": "Petals and hips are used for their cooling, anti-inflammatory properties, and as a source of Vitamin C."},
    "Santalum Album (Sandalwood)": {"is_medicinal": True, "uses": "Used externally for cooling skin inflammation and its aromatic, calming effects in aromatherapy."},
    "Saraca Asoca (Ashoka)": {"is_medicinal": True, "uses": "Revered in Ayurveda for its uterine tonic properties, used to treat gynecological disorders."},
    "Saraca Asoca (Seethaashoka)": {"is_medicinal": True, "uses": "Same medicinal properties as Ashoka, primarily used as a uterine tonic and for bleeding disorders."},
    "Solanum Lycopersicum (Tomato)": {"is_medicinal": True, "uses": "Rich in lycopene (antioxidant) and vitamins; consumed for overall health benefits."},
    "Solanum Nigrum (Black Nightshade)": {"is_medicinal": True, "uses": "Leaves are used externally for skin diseases; berries are used in liver and kidney remedies (Note: Parts can be toxic)."},
    "Spinacia Oleracea (Spinach)": {"is_medicinal": True, "uses": "Highly nutritious, rich in iron and vitamins, consumed for blood and bone health."},
    "Stereospermum Chelonoides (Padri)": {"is_medicinal": True, "uses": "Roots are part of the 'Dashamoola' formulation, used as an anti-inflammatory and nervine tonic."},
    "Syzygium Cumini (Jamun)": {"is_medicinal": True, "uses": "Seeds are traditionally used to help manage diabetes; fruit is beneficial for digestion."},
    "Syzygium Jambos (Rose Apple)": {"is_medicinal": True, "uses": "Used traditionally as a cooling agent and the seed powder is sometimes used for diarrhea."},
    "Tabernaemontana Divaricata (Crape Jasmine)": {"is_medicinal": True, "uses": "Roots and sap are used in folk medicine to treat eye diseases and skin problems."},
    "Tagetes (Marigold)": {"is_medicinal": True, "uses": "Used to treat skin inflammation, wounds, and for its antiseptic properties."},
    "Tamarindus Indica (Tamarind)": {"is_medicinal": True, "uses": "Fruit pulp is a mild laxative, and bark/leaves are used for wounds and inflammation."},
    "Tecoma Stans (Yellow Bells)": {"is_medicinal": True, "uses": "Used in traditional remedies for fever and digestive issues."},
    "Tinospora Cordifolia (Amruthaballi)": {"is_medicinal": True, "uses": "Powerful adaptogen and immune modulator, used to manage chronic fevers and improve digestion."},
    "Trigonella Foenum-graecum (Fenugreek)": {"is_medicinal": True, "uses": "Used to lower blood sugar, improve digestion, and stimulate milk production in nursing mothers."},
    "Turmeric": {"is_medicinal": True, "uses": "Contains curcumin, a potent anti-inflammatory and antioxidant; used widely for wound healing and immunity."},
    "Zingiber Officinale (Ginger)": {"is_medicinal": True, "uses": "Used to alleviate nausea, motion sickness, digestive problems, and reduce inflammation."},
    "kepala": {"is_medicinal": True, "uses": "Used in local remedies for pain relief and digestive health (Placeholder: Please verify and expand)."},
    "Non-Medicinal Sample": {"is_medicinal": False, "uses": "Not categorized as a primary Indian medicinal plant. Its use is primarily ornamental or structural."}
}

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app) 
model = None
# The global db_plant_data will be populated by load_model_and_data
db_plant_data = {} 


# --- MODEL ARCHITECTURE (Extracted from your notebook) ---

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class AttentionEfficientNet(nn.Module):
    def __init__(self, num_classes, model_name='tf_efficientnetv2_b3', pretrained=True, dropout=0.4):
        super().__init__()
        # Use forward_features only if needed, otherwise rely on default structure
        self.base_model = create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.num_features = self.base_model.num_features 

        self.attention = CBAM(self.num_features, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.attention(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- UTILITY FUNCTIONS ---

def get_inference_transform(img_size=IMG_SIZE):
    """
    Standardizes the inference transforms used in the original notebook.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model_and_data():
    """
    Load the PyTorch model and the plant uses database.
    """
    global model, db_plant_data
    print(f"Device: {DEVICE}. Loading model from: {MODEL_PATH}...")
    
    try:
        # Load the database dictionary globally
        db_plant_data.update(DB_PLANT_DATA)
        
        model = AttentionEfficientNet(num_classes=NUM_CLASSES, model_name=MODEL_NAME, pretrained=False)
        
        # Load state dictionary
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Extract model state dict from the checkpoint structure
        model_state_dict = checkpoint.get('model_state_dict', checkpoint) 
        model.load_state_dict(model_state_dict)
        model.to(DEVICE)
        model.eval() 
        print("PyTorch Model loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model or data: {e}. Please ensure PyTorch is installed and {MODEL_PATH} exists.")
        model = None 

def preprocess_and_remove_background(image_bytes):
    """
    Performs image decoding, background removal using rembg, and conversion to RGB with white background.
    """
    
    img = Image.open(io.BytesIO(image_bytes))
    
    # 1. Background Removal
    # We use a session with the 'u2net' model which is suitable for general objects/plants.
    processed_img_rgba = remove(img, post_process_mask=True) 
    
    # 2. Convert RGBA to RGB with white background (as done in your notebook)
    if processed_img_rgba.mode == 'RGBA':
        background = Image.new('RGB', processed_img_rgba.size, (255, 255, 255))
        background.paste(processed_img_rgba, mask=processed_img_rgba.split()[3])
        processed_img_pil = background
    elif processed_img_rgba.mode != 'RGB':
        processed_img_pil = processed_img_rgba.convert('RGB')
    else:
        processed_img_pil = processed_img_rgba

    # 3. Convert processed image to base64 (for frontend display)
    buffered = io.BytesIO()
    processed_img_pil.save(buffered, format="PNG") 
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Return the processed PIL image for classification and its base64 string
    return processed_img_pil, f"data:image/png;base64,{img_b64}"


def run_inference(processed_image_pil):
    """
    Runs classification inference using the loaded PyTorch model.
    """
    if model is None:
        raise RuntimeError("AI Model not loaded.")
        
    transform = get_inference_transform()
    
    # 1. Convert PIL image to PyTorch tensor
    img_tensor = transform(processed_image_pil).unsqueeze(0).to(DEVICE)

    # 2. Run inference
    with torch.no_grad():
        # Use autocast for mixed precision inference if CUDA is available
        with torch.cuda.amp.autocast():
            output = model(img_tensor)
            
        probs = torch.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probs, 1)
        
        # Get the class name using the index
        plant_name = CLASS_NAMES[predicted_index.item()]
        confidence_pct = confidence.item() * 100
        
        return plant_name, round(confidence_pct, 2)

# --- API ENDPOINTS ---

@app.route('/classify', methods=['POST'])
def classify_plant():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        
        # 1. Background Removal and Preprocessing
        processed_image, b64_image_url = preprocess_and_remove_background(image_bytes)
        
        # 2. Classification Inference
        plant_name, confidence = run_inference(processed_image)
        
        # 3. Fetch data from the database
        plant_data = db_plant_data.get(plant_name, db_plant_data.get("Non-Medicinal Sample"))
        
        # Final decision on medicinal status (if the name is found in the database and marked medicinal)
        is_medicinal = plant_data.get('is_medicinal', False)
        
        response_data = {
            "name": plant_name,
            "is_medicinal": is_medicinal,
            "uses": plant_data['uses'],
            "confidence": f"{confidence}%",
            "processed_image_url": b64_image_url,
        }
        
        return jsonify(response_data)
        
    except RuntimeError as e:
        print(f"Classification Runtime Error: {e}")
        return jsonify({"error": f"Model Error: {str(e)}. Please check model loading and PyTorch environment."}), 500
    except Exception as e:
        print(f"Classification General Error: {e}")
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat_bot():
    data = request.get_json()
    query = data.get('query')
    plant_name = data.get('plant_name') 

    if not query or not plant_name:
        return jsonify({"error": "Missing query or plant_name in request"}), 400

    # --- ENHANCED SYSTEM INSTRUCTION ---
    # This prompt strictly enforces domain-specific, appropriate answers.
    system_prompt = (
        f"You are a highly specialized AI assistant for Indian medicinal plants. "
        f"The current user context is the identified plant: {plant_name}. "
        f"Your role is to provide factual, informative, and concise answers based on scientific and traditional knowledge. "
        f"Your knowledge is strictly limited to the properties, uses, common dosage forms, traditional preparations, cultivation details, and identification "
        f"details of Indian medicinal plants, especially the {NUM_CLASSES} classes in our database. "
        f"\n\n**CRITICAL RESPONSE RULES:**"
        f"\n1. **Maintain Domain Focus (Vague/Out-of-Scope Questions):** If the user asks a question outside the scope of Indian medicinal plants "
        f"(e.g., general history, finance, non-Indian plants, medical diagnosis, or botany unrelated to medicinal use), you **MUST politely refuse** and state: 'I am only programmed to answer questions about Indian medicinal plants. Please ask me a question related to {plant_name} or another plant in our domain.' "
        f"\n2. **Avoid Medical Advice:** You **MUST NOT** provide definitive medical diagnoses, prescriptions, or advice on replacing conventional medicine. Use cautious language: 'Traditional use suggests...', 'Consult a healthcare professional for...', or 'This information is for knowledge only.'"
        f"\n3. **Provide Context:** Always reference the plant name ({plant_name}) in your answer where relevant."
        f"\n4. **Handling Vague Queries:** If a query is too vague (e.g., 'tell me about medicine'), ask for clarification, such as: 'Could you please specify which plant or which aspect of medicinal use you are interested in?'"
    )
    # --- END ENHANCED SYSTEM INSTRUCTION ---
    
    payload = {
        "contents": [{"parts": [{"text": query}]}],
        "tools": [{"google_search": {} }], # Use Google Search for grounded, up-to-date facts
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # Using exponential backoff for robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'AI bot response failed.')
                return jsonify({"response": generated_text})
            
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt) 
            else:
                response.raise_for_status() 

        except requests.exceptions.RequestException as e:
            print(f"Gemini API request failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return jsonify({"error": "Failed to communicate with the AI model. Check API Key and network."}), 500
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return jsonify({"error": "Internal server error during chat."}), 500

# Run the model loading when the server starts
load_model_and_data()

if __name__ == '__main__':
    # host='0.0.0.0' allows the server to be accessed by other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=True)