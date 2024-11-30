from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
import base64
from pydantic import BaseModel
import tempfile
import io

app = FastAPI(title="Plant Disease Detection API")

# Add CORS middleware
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
# Public S3 URL for the model
MODEL_S3_URL = "https://college-101.s3.ap-south-1.amazonaws.com/model.keras"
model = None

# PlantVillage dataset class names
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def download_model(url: str) -> str:
    """Download model from a public S3 URL."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download model. HTTP status: {response.status_code}")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

@app.on_event("startup")
async def load_model():
    """Load the model when the application starts."""
    global model
    try:
        model_path = download_model(MODEL_S3_URL)
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def clean_class_name(class_name):
    """Convert class name to a more readable format"""
    plant, condition = class_name.split('___')
    condition = condition.replace('_', ' ')
    return f"{plant} - {condition}"

def preprocess_image(image_bytes):
    """Preprocess the image bytes for model prediction"""
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to match model's expected sizing
    image = image.resize((224, 224))

    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array / 255.0

    # Add batch dimension
    return np.expand_dims(image_array, axis=0)

def generate_prediction(processed_image):
    """Generate prediction from processed image"""
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    raw_class_name = CLASS_NAMES[predicted_class]

    return {
        "success": True,
        "predictions": {
            "class": {
                "raw": raw_class_name,
                "clean": clean_class_name(raw_class_name)
            },
            "confidence": confidence,
            "detailed_predictions": {
                clean_class_name(CLASS_NAMES[i]): float(predictions[0][i])
                for i in range(len(CLASS_NAMES))
            }
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict plant disease from image file",
            "/predict-url": "POST - Predict plant disease from image URL",
            "/predict-data-url": "POST - Predict plant disease from base64-encoded image data",
            "/classes": "GET - List all possible classes"
        }
    }

@app.get("/classes")
async def get_classes():
    """Return all possible classes with both raw and clean formats"""
    return {
        "classes": [
            {
                "raw": class_name,
                "clean": clean_class_name(class_name)
            }
            for class_name in CLASS_NAMES
        ]
    }

class ImageUrl(BaseModel):
    image_url: str

class ImageDataUrl(BaseModel):
    data_url: str

@app.post("/predict-url")
async def predict_from_url(request: ImageUrl):
    """
    Predict plant disease from image URL

    Parameters:
    - request: JSON containing image_url
    """
    try:
        # Download image from URL
        response = requests.get(request.image_url)
        image_bytes = response.content

        # Preprocess the image
        processed_image = preprocess_image(image_bytes)

        # Generate prediction
        return generate_prediction(processed_image)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-data-url")
async def predict_from_data_url(request: ImageDataUrl):
    """
    Predict plant disease from base64-encoded image data

    Parameters:
    - request: JSON containing data_url
    """
    try:
        # Extract base64 image data from the data URL
        image_data = request.data_url.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Preprocess the image
        processed_image = preprocess_image(image_bytes)

        # Generate prediction
        return generate_prediction(processed_image)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image file

    Parameters:
    - file: Uploaded image file
    """
    try:
        # Read image file
        image_bytes = await file.read()

        # Preprocess the image
        processed_image = preprocess_image(image_bytes)

        # Generate prediction
        return generate_prediction(processed_image)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)