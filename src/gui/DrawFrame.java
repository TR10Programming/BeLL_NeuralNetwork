/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gui;

import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.math.NeuralMath;

/**
 *
 * @author Felix
 */
public class DrawFrame extends javax.swing.JFrame {

    private NeuralNetwork nn;
    
    /**
     * Creates new form DrawFrame
     * @param nn
     */
    public DrawFrame(NeuralNetwork nn) {
        this.nn = nn;
        initComponents();
        getDrawPanel().setOnUpdateHandler(this::updatePredictions);
    }
    
    public final DrawPanel getDrawPanel() {
        return (DrawPanel) drawPanel;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        drawPanel = new DrawPanel();
        btnClear = new javax.swing.JButton();
        lblPredictedClass = new javax.swing.JLabel();
        lblClasses = new javax.swing.JLabel();
        lblPredictedClassProb = new javax.swing.JLabel();
        btnClose = new javax.swing.JButton();
        slThickness = new javax.swing.JSlider();
        jLabel1 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle("Zeichnen");

        javax.swing.GroupLayout drawPanelLayout = new javax.swing.GroupLayout(drawPanel);
        drawPanel.setLayout(drawPanelLayout);
        drawPanelLayout.setHorizontalGroup(
            drawPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 280, Short.MAX_VALUE)
        );
        drawPanelLayout.setVerticalGroup(
            drawPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 280, Short.MAX_VALUE)
        );

        btnClear.setText("Zeichenfläche leeren");
        btnClear.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnClearActionPerformed(evt);
            }
        });

        lblPredictedClass.setFont(new java.awt.Font("Tahoma", 0, 24)); // NOI18N
        lblPredictedClass.setText("-");

        lblClasses.setText("Keine Klassen festgelegt");
        lblClasses.setVerticalAlignment(javax.swing.SwingConstants.TOP);

        lblPredictedClassProb.setText(" 0,0%");

        btnClose.setText("Schließen");
        btnClose.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnCloseActionPerformed(evt);
            }
        });

        slThickness.setMajorTickSpacing(1);
        slThickness.setMaximum(3);
        slThickness.setMinimum(1);
        slThickness.setPaintTicks(true);
        slThickness.setSnapToTicks(true);
        slThickness.setValue(2);
        slThickness.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                slThicknessStateChanged(evt);
            }
        });

        jLabel1.setText("Zeichenstärke:");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(99, 99, 99)
                        .addComponent(slThickness, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(drawPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 85, javax.swing.GroupLayout.PREFERRED_SIZE))))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(btnClear)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(btnClose))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(lblPredictedClass)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(lblPredictedClassProb))
                            .addComponent(lblClasses, javax.swing.GroupLayout.PREFERRED_SIZE, 276, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(drawPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(lblPredictedClass)
                            .addComponent(lblPredictedClassProb))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(lblClasses, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(btnClear)
                            .addComponent(btnClose))))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(slThickness, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jLabel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap(33, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void btnCloseActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnCloseActionPerformed
        dispose();
    }//GEN-LAST:event_btnCloseActionPerformed

    private void btnClearActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnClearActionPerformed
        getDrawPanel().clear();
    }//GEN-LAST:event_btnClearActionPerformed

    private void slThicknessStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_slThicknessStateChanged
        getDrawPanel().setThickness(slThickness.getValue());
    }//GEN-LAST:event_slThicknessStateChanged

    public void updatePredictions(double[][] data) {
        double[] out = nn.trigger(NeuralMath.flatten(data)).getOutput();
        int prediction = NeuralMath.getPredictedLabel(out);
        lblPredictedClass.setText(prediction + "");
        lblPredictedClassProb.setText(((int) (out[prediction] * 100)) + "%");
        String allClasses = "<html>";
        for(int i = 0; i < out.length; i++) {
            allClasses += i + ": " + ((int) (out[i] * 100)) + "%<br>";
        }
        allClasses += "</html>";
        lblClasses.setText(allClasses);
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnClear;
    private javax.swing.JButton btnClose;
    private javax.swing.JPanel drawPanel;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel lblClasses;
    private javax.swing.JLabel lblPredictedClass;
    private javax.swing.JLabel lblPredictedClassProb;
    private javax.swing.JSlider slThickness;
    // End of variables declaration//GEN-END:variables
}
