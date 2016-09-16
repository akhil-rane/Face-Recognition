package com.floatinvoice;

import java.io.File;
import java.nio.IntBuffer; 

import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacpp.helper.opencv_core.CvArr;

import static org.bytedeco.javacpp.opencv_face.*; 
import static org.bytedeco.javacpp.opencv_core.*; 
import static org.bytedeco.javacpp.opencv_imgcodecs.*; 
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class FaceRecognizer {
    
	public static void main(String[] args) {
        
		String trainingDir = "E:/faces/training/";
                
        Mat testImage = imread("E:/faces/faces/image_0020.jpg", CV_LOAD_IMAGE_GRAYSCALE);

        File root = new File(trainingDir);        
               
        File[] imageFiles = root.listFiles();

        MatVector images = new MatVector(imageFiles.length);

        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        
        IntBuffer labelsBuf = labels.createBuffer();

        CascadeClassifier faceDetector=new CascadeClassifier();
        
        faceDetector.load("E:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
        
        RectVector faceDetections = new RectVector();
        
        int count = 0;
        
        for (File image : imageFiles) {
            
        	Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);         
                		
            faceDetector.detectMultiScale(img, faceDetections);           
                     
            Mat croppedFace=new Mat(img,faceDetections.get(0l));          
           
           
          
            
            int label = Integer.parseInt(image.getName().split("-")[0]);

            images.put(count, croppedFace);
            
            labelsBuf.put(count, label);

            count++;  
        }
        
        BasicFaceRecognizer faceRecognizer = createFisherFaceRecognizer();

        faceRecognizer.train(images, labels);

        faceDetector.detectMultiScale(testImage, faceDetections);           
        
        Mat testImageFinal = new Mat(testImage,faceDetections.get(0l)); 
        
        int predictedLabel = faceRecognizer.predict(testImageFinal);

        System.out.println("Predicted label: " + predictedLabel);
    }
}
