package gui;

import de.fk.neuralnetwork.data.ImageContainer;
import de.fk.neuralnetwork.data.LabeledImage;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.List;
import javax.swing.JPanel;

/**
 *
 * @author Felix
 */
public class TrainingExampleDisplayPanel extends JPanel {
    
    public static final Font FONT = new Font("Arial", Font.PLAIN, 12);
    public static final int IMAGE_SCALE = 2, PADDING = 20, SPACING = 10, SPACING_BOTTOM = 30;

    public TrainingExampleDisplayPanel() {
        super(true);
    }

    @Override
    public void paint(Graphics grphcs) {
        Graphics2D g = (Graphics2D) grphcs;
        int pw = getWidth(), ph = getHeight();
        g.setFont(FONT);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, pw, ph);
        List<LabeledImage> imgs = ImageContainer.getImages();
        int x = PADDING, y = PADDING, maxHeight = 0;
        for(int i = 0; i < imgs.size(); i++) {
            LabeledImage cimg = imgs.get(i);
            double[][] cdata = cimg.getData();
            //Überprüfe ob Zeilenumbruch notwendig
            int cy = IMAGE_SCALE * cdata.length + SPACING_BOTTOM, cx = IMAGE_SCALE * cdata[0].length + SPACING;
            if(x + cx > pw - PADDING) {//Zeilenumbruch
                x = PADDING;
                y += maxHeight;
                maxHeight = 0;
            }
            if(y + cy > ph - PADDING) { //Panel ist voll; Zeichnen fertig
                break;
            }
            //Bild zeichnen
            for(int row = 0; row < cdata.length; row++) {
                for(int col = 0; col < cdata[row].length; col++) {
                    g.setColor(new Color(0f, 0f, 0f, (float) cdata[row][col]));
                    g.fillRect(x + col * IMAGE_SCALE, y + row * IMAGE_SCALE, IMAGE_SCALE, IMAGE_SCALE);
                }
            }
            //Label
            g.setColor(Color.BLACK);
            g.drawString("[" + cimg.getLabel() + "]", x + cx / 2 - 6, y + cy - 6);
            x += cx; //nach rechts bewegen
            if(maxHeight < cy) maxHeight = cy; //maxHeight updaten
        }
    }
    
}
