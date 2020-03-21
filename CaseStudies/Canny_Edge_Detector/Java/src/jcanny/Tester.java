package jcanny;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/**
 * This class demonstrates the usage of the JCanny Canny edge detector library.
 * 
 * @author Levon
 */

public class Tester {
    //Canny parameters
    private static final double CANNY_THRESHOLD_RATIO = .2; //Suggested range .2 - .4
    private static final int CANNY_STD_DEV = 1;             //Range 1-3
    
    //I/O parameters
    private static String imgFileName;
    private static String imgOutFile = "";
    private static String imgExt;

    public static void main(String[] args) {
        //Read input file name and create output file name
        imgFileName = args[0];
        imgExt = args[1];
        String[] arr = imgFileName.split("\\.");
        
        for (int i = 0; i < arr.length - 1; i++) {
            imgOutFile += arr[i];
        }
        
        imgOutFile += "_canny.";
        imgOutFile += imgExt;
        
        //Sample JCanny usage
        try {
            BufferedImage input = ImageIO.read(new File(imgFileName));
            BufferedImage output = JCanny.CannyEdges(input, CANNY_STD_DEV, CANNY_THRESHOLD_RATIO);
            ImageIO.write(output, imgExt, new File(imgOutFile));
        } catch (Exception ex) {
            System.out.println("ERROR ACCESING IMAGE FILE:\n" + ex.getMessage());
        }
    }    
}
