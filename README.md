# Face-Recognition-and-Optical-Character-Recognition
This study is divided into two parts: Task 1 - Face recognition and Task 2 - Optical Character Recognition. In this study, a database of 69 people has been considered. The database consists of individual images of 69 people, each holding a number between 1 and 81. The database consisted of around 5 still images and 5 live images (videos), each of 69 people. To build a larger database for more efficient training of models, 5 extra frames were extracted from each of the videos. Thus, for every person, around 25-30 images were available for training and testing. In Task 1, the objective was to identify the faces (face recognition) of people (present in the database) in an unseen group image. The Task 2 requires the identification of the number that each person is holding through Optical Character Recognition (OCR). These tasks were achieved using the Computer Vision Toolkit of MATLAB R2018b.


The folder consists of following
folders and sub-folders:
1. Face recognition
1.1. m files – contains ‘.m’ files for 5 classifier models, namely Hog-SVM,
Bag-SVM, Hog-MLP, Bag-MLP and CNN, one Emotion Classifier
file, and RecogniseFace function file.
RecogniseFace.m should be used by you to detect faces in any
image and returns the P matrix.
1.2. mat files – contains ‘.mat’ files for the 5 classifier models, namely
Hog-SVM, Bag-SVM, Hog-MLP, Bag-MLP and CNN, one Emotion
Classifier file, and a confusion matrix for Hog-SVM.
These ‘.mat’ files should be loaded while calling the RecogniseFace
function.
2. OCR
It contains two ‘.m’ files named OCR and DetectNum.
OCR.m contains the code for optical character recognition, and
DetectNum.m is the function that should be used by you to detect digits in any
image.
