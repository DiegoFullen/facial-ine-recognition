# Facial verification with INE using Facenet and PyTorch
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C)
![FaceNet](https://img.shields.io/badge/FaceNet-PyTorch-informational)
![NumPy](https://img.shields.io/badge/NumPy-1.26-lightgrey)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-yellowgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-red)

This project performs facial identity verification by comparing a normal face image (selfie) with the photo on a government-issued ID (INE card). It uses `MTCNN` for face detection and `InceptionResnetV1` for embedding extraction, based on the `facenet-pytorch` library.

## Project Structure

/emotion-recognition-mediapipe-knn/  
  │  
  ├── datasets/               # Main folder for CSV  
      ├── same/               # Contains pairs of images of the same person  
      ├── different/          # Contains pairs of images of the different person  
  ├── metrics/                # Model performance images 
  ├── models/                 # Contains the trained model  
  ├── photo_test/             # Pairs of photos for masive test  
  ├── training/               # Training and use of the model functions
  ├── utils/                  # Creation dataset function  
  ├── main.py                 # Entry point  
  ├── requirements.txt        # Dependencies  
  └── README.md  

# Note
The `/same`, `/different` folder contains pairs of images of each person, where each pair is composed of:
  - `persona_01_foto.jpg`: Image of the person (can be a selfie).
  - `persona_01_ine.jpg`: Image of the identity document (INE) of the same person.

## How to Run
### 1. Clone repository
```bash
git clone https://github.com/DiegoFullen/facial-ine-recognition.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the system
```bash
python main.py
```


## Technical Stack
Core libraries used:
- Deep Learning: PyTorch, facenet-pytorch
- Machine Learning: scikit-learn
- Computer Vision: OpenCV, PIL
- Data Handling: NumPy, Pickle
- Visualization: matplotlib, seaborn
- Face Detection and Embedding: MTCNN, InceptionResnetV1 (pretrained on VGGFace2)


## Technologies used
- Python 3.8+
- PyTorch
- facenet-pytorch
- OpenCV
- NumPy
- PIL (Pillow)


## Notes
- This project was developed for educational purposes.
- No personal data is used or stored.
- No GUI implemented yet – console interface only.
- The INE photo may require preprocessing if it's noisy or low contrast.
*Specific versions pinned for reproducibility (see requirements.txt)*


## Project Context
Schoolar graduation project developed for EstacionaT team - CETI Colomos  
*Not intended for commercial use* 


## Author
Diego Salvador Candia Fullen
