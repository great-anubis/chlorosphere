import requests

# Flask API URL (make sure your Flask server is running)
UPLOAD_URL = "http://127.0.0.1:5000/upload"

# Paths to the test images
TEST_IMAGES = [
    "../tests/test_sample_1.png",
    "../tests/test_sample_2.png",
    "../tests/test_sample_3.png",
    "../tests/test_sample_4.png",
]

def test_upload(image_path):
    """
    Test the `/upload` endpoint by uploading an image file.
    """
    print(f"Testing upload with file: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(UPLOAD_URL, files=files)
        
        if response.status_code == 200:
            print("Upload successful! Response:")
            print(response.json())
        else:
            print(f"Failed to upload. Status code: {response.status_code}")
            print("Response:", response.text)

    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Iterate through the test images and test the upload
    for image_path in TEST_IMAGES:
        test_upload(image_path)

PREDICT_URL = "http://127.0.0.1:5000/predict"

def test_predict(file_path):
    """
    Test the `/predict` endpoint using the uploaded file path.
    """
    print(f"Testing prediction for file: {file_path}")
    payload = {"file_path": file_path}

    try:
        response = requests.post(PREDICT_URL, json=payload)
        if response.status_code == 200:
            print("Prediction successful! Response:")
            print(response.json())
        else:
            print(f"Failed to predict. Status code: {response.status_code}")
            print("Response:", response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Test both upload and predict
    for image_path in TEST_IMAGES:
        test_upload(image_path)
        test_predict(image_path)
