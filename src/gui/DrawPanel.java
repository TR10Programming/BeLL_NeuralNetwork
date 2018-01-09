package gui;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.function.Consumer;
import javax.swing.JPanel;
import static java.lang.Math.min;

/**
 *
 * @author Felix
 */
public class DrawPanel extends JPanel implements MouseListener, MouseMotionListener {

    public static final int WIDTH = 28, HEIGHT = 28;
    
    private double[][] data = new double[WIDTH][HEIGHT];
    private Consumer<double[][]> onUpdateHandler = null;
    private int thickness = 2;
    
    public DrawPanel() {
        super(true);
        super.addMouseListener(this);
        super.addMouseMotionListener(this);
    }

    public void setOnUpdateHandler(Consumer<double[][]> onUpdateHandler) {
        this.onUpdateHandler = onUpdateHandler;
    }
    
    public void clear() {
        data = new double[WIDTH][HEIGHT];
        repaint();
        if(onUpdateHandler != null) onUpdateHandler.accept(data);
    }

    public void setThickness(int thickness) {
        this.thickness = thickness;
    }

    @Override
    public void paint(Graphics g) {
        int w = getWidth() / WIDTH, h = getHeight() / HEIGHT;
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, getWidth(), getHeight());
        for(int y = 0; y < HEIGHT; y++)
            for(int x = 0; x < WIDTH; x++) {
                g.setColor(new Color(0, 0, 0, (float) data[y][x]));
                g.fillRect(x * w, y * h, w, h);
            }
    }

    private void mouseEvent(MouseEvent e) {
        int x = (int) (((double) e.getX()) / ((double) getWidth()) * WIDTH),
                y = (int) (((double) e.getY()) / ((double) getHeight()) * HEIGHT);
        if(x >= 0 && y >= 0 && x < WIDTH && y < HEIGHT) {
            data[y][x] = min(1, data[y][x] + 0.5);
            if(thickness >= 2) {
                data[y - 1][x] = min(1, data[y - 1][x] + 0.2);
                data[y + 1][x] = min(1, data[y + 1][x] + 0.2);
                data[y][x - 1] = min(1, data[y][x - 1] + 0.2);
                data[y][x + 1] = min(1, data[y][x + 1] + 0.2);
                if(thickness >= 3) {
                    data[y - 2][x] = min(1, data[y - 2][x] + 0.1);
                    data[y + 2][x] = min(1, data[y + 2][x] + 0.1);
                    data[y][x - 2] = min(1, data[y][x - 2] + 0.1);
                    data[y][x + 2] = min(1, data[y][x + 2] + 0.1);
                    data[y - 1][x - 1] = min(1, data[y - 1][x - 1] + 0.05);
                    data[y + 1][x - 1] = min(1, data[y + 1][x - 1] + 0.05);
                    data[y + 1][x + 1] = min(1, data[y + 1][x + 1] + 0.05);
                    data[y - 1][x + 1] = min(1, data[y - 1][x + 1] + 0.05);
                }
            }
            repaint();
        }
        if(onUpdateHandler != null) onUpdateHandler.accept(data);
    }
    
    @Override
    public void mousePressed(MouseEvent e) { mouseEvent(e); }

    @Override
    public void mouseDragged(MouseEvent e) { mouseEvent(e); }
    
    //Unused
    @Override
    public void mouseClicked(MouseEvent e) { }
    @Override
    public void mouseEntered(MouseEvent e) { }
    @Override
    public void mouseExited(MouseEvent e) { }
    @Override
    public void mouseMoved(MouseEvent e) { }
    @Override
    public void mouseReleased(MouseEvent e) { }
    
}
