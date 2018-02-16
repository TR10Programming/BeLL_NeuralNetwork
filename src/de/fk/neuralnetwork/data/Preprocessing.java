package de.fk.neuralnetwork.data;

/**
 *
 * @author Felix
 */
public class Preprocessing {

    public static final double[][] rotate(double[][] data, double rad) {
        int height = data.length, width = data[0].length;
        double sin = Math.sin(rad), cos = Math.cos(rad),
                x0 = (width - 1) / 2.0, y0 = (height - 1) / 2.0;
        double[][] out = new double[height][width];
        for(int y = 0; y < height; y++)
            for(int x = 0; x < width; x++) {
                //Abstände zum Mittelpunkt a,b
                double a = x - x0, b = y - y0;
                //Koordinaten des Punktes vor Rotation
                double xn = a * cos - b * sin + x0,
                        yn = a * sin + b * cos + y0;
                //Ganzzahliger Anteil
                int xni = (int) xn, yni = (int) yn;
                //Gleitkommaanteil/Abweichung von ganzen Pixeln
                double xu = Math.max(0, xn - xni),
                        yu = Math.max(0, yn - yni);
                //Anteilweises Projizieren des Pixels und seiner Nachbarpixel auf den neuen Punkt x,y
                //Grundbedingung (sonst alle Pixel außerhalb des Bildes)
                if(yni >= -1 && xni >= -1 && xni < width && yni < height) {
                    //Bedingung neg. Y
                    if(yni >= 0) {
                        //Benachbarter Pixel neg. Y/pos. X
                        if(xni + 1 < width) out[y][x] += (1 - yu) * xu * data[yni][xni + 1];
                        //Benachbarter Pixel neg. Y/neg. X
                        if(xni >= 0) out[y][x] += (1 - yu) * (1 - xu) * data[yni][xni];
                    }
                    //Bedingung pos. Y
                    if(yni + 1 < height) {
                        //Benachbarter Pixel pos. Y/pos. X
                        if(xni + 1 < width) out[y][x] += yu * xu * data[yni + 1][xni + 1];
                        //Benachbarter Pixel pos. Y/neg. X
                        if(xni >= 0) out[y][x] += yu * (1 - xu) * data[yni + 1][xni];
                    }
                }
            }
        return out;
    }
    
    public static final double[][] scale(double[][] data, double fact) {
        int height = data.length, width = data[0].length;
        double x0 = (width - 1) / 2.0, y0 = (height - 1) / 2.0;
        double[][] out = new double[height][width];
        for(int y = 0; y < height; y++)
            for(int x = 0; x < width; x++) {
                int xn = (int) (x0 + (x - x0) / fact),
                        yn = (int) (y0 + (y - y0) / fact);
                if(xn >= 0 && xn < width && yn >= 0 && yn < height)
                    out[y][x] = data[yn][xn];
            }
        return out;
    }
    
    public static final double[][] shift(double[][] data, int dx, int dy) {
        int height = data.length, width = data[0].length;
        int ystart = Math.max(0, dy), yend = height + Math.min(0, dy),
                xstart = Math.max(0, dx), xend = width + Math.min(0, dx);
        double[][] out = new double[height][width];
        for(int y = ystart; y < yend; y++)
            for(int x = xstart; x < xend; x++)
                out[y][x] = data[y - dy][x - dx];
        return out;
    }
    
}
