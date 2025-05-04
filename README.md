# Facial Verification with INE (ID) using FaceNet and PyTorch
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C)
![FaceNet](https://img.shields.io/badge/FaceNet-PyTorch-informational)
![NumPy](https://img.shields.io/badge/NumPy-1.26-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-red)

This project performs facial identity verification by comparing a normal face image (selfie) with the photo on a government-issued ID (INE card). It uses `MTCNN` for face detection and `InceptionResnetV1` for embedding extraction, based on the `facenet-pytorch` library.

## Project Structure

/facial-ine-recognition/  
  ├── datasets/               # Main folder for CSV  
      ├── same/               # Pairs of the same person (selfie + INE)  
      ├── different/          # Pairs of different people (selfie + INE)  
  ├── metrics/                # Model performance visuals 
  ├── models/                 # Saved trained model (.pkl)  
  ├── photo_test/             # Image pairs for model inference  
  ├── training/               # Training and inference scripts  
  ├── utils/                  # Helper functions (e.g., dataset creation)  
  ├── main.py                 # Entry point  
  └── requirements.txt        # Project dependencies  

# Notes for Images
The `/same` and `/different` directories contain image pairs:
  - `persona_01_foto.jpg`: Regular face image (e.g., a selfie).
  - `persona_01_ine.jpg`: Image extracted from the person's INE (ID card).
The `photo_test/` folder is used for bulk inference testing (not training or validation).

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
- Data Handling: NumPy, Pickle, OS
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
Scholarly graduation project developed for EstacionaT team - CETI Colomos  
*Not intended for commercial use* 


## Author
Diego Salvador Candia Fullen
