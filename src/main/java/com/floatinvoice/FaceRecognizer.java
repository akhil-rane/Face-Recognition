package com.floatinvoice;

import java.io.File;
import java.nio.IntBuffer; 

import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;

import static org.bytedeco.javacpp.opencv_face.*; 
import static org.bytedeco.javacpp.opencv_core.*; 
import static org.bytedeco.javacpp.opencv_imgcodecs.*; 


public class FaceRecognizer {

	public static void main(String[] args) {

		//Initializing training set
		String trainingDir = "Test_Dataset/sample_training_set/";

		File trainRoot = new File(trainingDir);        

		File[] imageFiles = trainRoot.listFiles();

		MatVector images = new MatVector(imageFiles.length);

		Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);

		IntBuffer labelsBuf = labels.createBuffer();

		//Initializing test set
		String testDir = "Test_Dataset/sample_test_set/";

		File testRoot = new File(testDir);

		File[] testFiles = testRoot.listFiles();

		MatVector testImages = new MatVector(testFiles.length);

		Mat testLabels = new Mat(testFiles.length, 1, CV_32SC1);

		//Training face detection and recognition model  
		CascadeClassifier faceDetector=new CascadeClassifier();

		faceDetector.load("Resources/haarcascade_frontalface_alt.xml");

		RectVector faceDetections = new RectVector();

		int count = 0;

		for (File image : imageFiles) {


			Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);         

			faceDetector.detectMultiScale(img, faceDetections);           

			Mat croppedFace=new Mat(img,faceDetections.get(0l));          

			int label = Integer.parseInt(image.getName().split("-")[0]);

			images.put(count,croppedFace);

			labelsBuf.put(count, label);

			count++;  
		}

		LBPHFaceRecognizer faceRecognizer = createLBPHFaceRecognizer();

		faceRecognizer.train(images, labels);

		
		//Test set Prediction
		for (File image : testFiles) {

			Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);         

			faceDetector.detectMultiScale(img, faceDetections);           

			Mat croppedFace=new Mat(img,faceDetections.get(0l));          

			int actualLabel = Integer.parseInt(image.getName().split("-")[0]);
			
			System.out.println("Actual label: " + actualLabel);
			
			int predictedLabel = faceRecognizer.predict(croppedFace);        

			System.out.println("Predicted label: " + predictedLabel);
			
			System.out.println("--------------------------------------------------");	
		}

	}
}
