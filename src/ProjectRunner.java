import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class ProjectRunner {
    public static SequentialNet net;

    public static void main(String args[]) {
        net = new SequentialNet();
        net.setInputDim(784);
        net.setOutputDim(10);
        net.addHiddenLayer(48);
        net.addHiddenLayer(24);
        net.addHiddenLayer(24);

        String path = String.format("pixil-frame-0.png");

        // gets the image from the file
        File img = new File(path);
        BufferedImage bImg = new BufferedImage(28, 28, BufferedImage.TYPE_INT_ARGB);


    }

    public static void train() {

    }



}
