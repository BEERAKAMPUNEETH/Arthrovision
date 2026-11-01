import requests
import os

def test_inference():
    print("=" * 60)
    print("RUNNING ONE END-TO-END INFERENCE ON THE DEPLOYED BACKEND API")
    print("=" * 60)
    
    url = "http://127.0.0.1:8000/predict"
    image_path = "C:/Users/punee/Downloads/archive/KneeXrayMini/test/0/9003175L.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found at {image_path}")
        return
        
    print(f"Uploading X-ray image: {image_path}")
    with open(image_path, "rb") as img_file:
        files = {"image": (os.path.basename(image_path), img_file, "image/png")}
        try:
            response = requests.post(url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("\n[SUCCESS] API Response Received:")
                print(f"  Predicted Class: {result['predicted_class']}")
                print(f"  Predicted Grade: {result['predicted_grade']}")
                print("\n  Class Probabilities:")
                for idx, prob in enumerate(result['probabilities']):
                    print(f"    KL Grade {idx}: {prob:.2%}")
                
                print("\n  Explainable AI (Base64 Outputs):")
                print(f"    Original Image (b64 string length):      {len(result['original_image_b64'])} chars")
                print(f"    Grad-CAM Heatmap overlay (b64 length):  {len(result['grad_cam_b64'])} chars")
                print(f"    TDA Persistence Diagram (b64 length):   {len(result['persistence_diagram_b64'])} chars")
                print(f"    TDA Persistence Barcode (b64 length):   {len(result['persistence_barcode_b64'])} chars")
                
                print("\n  Clinical Modality Status:")
                print(f"    Clinical modality disabled: {result['clinical_modality_disabled']}")
            else:
                print(f"\n[ERROR] API request failed with status code {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("\n[ERROR] Could not connect to API server at http://127.0.0.1:8000. Ensure Uvicorn is active!")

if __name__ == "__main__":
    test_inference()
