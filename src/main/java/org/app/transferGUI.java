package org.app;

import javax.swing.*;
import javax.swing.plaf.nimbus.NimbusLookAndFeel;
import java.awt.*;
import java.awt.image.BufferedImage;

public class transferGUI extends JFrame {
    transferGUI () {
        try
        {
            UIManager.setLookAndFeel(new NimbusLookAndFeel());
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.out.println("Nimbus Look And Feel not supported");
        }
        setTitle("Style Transfer Demo Application");
        setSize(600,600);



        setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();


        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.anchor = GridBagConstraints.CENTER;

        JLabel appLabel = new JLabel("Style Transfer Demo Application");
        add(appLabel, gbc);



        //PlaceHolder Image
        BufferedImage placeholder = new BufferedImage(200, 200, BufferedImage.TYPE_INT_ARGB);

        // Draw something on the image
        Graphics2D g2d = placeholder.createGraphics();
        g2d.setPaint(Color.LIGHT_GRAY);
        g2d.fillRect(0, 0, placeholder.getWidth(), placeholder.getHeight());
        g2d.setPaint(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 6));
        g2d.drawString("Placeholder", 25, 25);
        g2d.dispose();

        // Components that will display the selected style and content images
        JLabel contentLabel = new JLabel(new ImageIcon(placeholder));
        JLabel styleLabel = new JLabel(new ImageIcon(placeholder));
        JPanel imagePanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 0));
        imagePanel.add(contentLabel);
        imagePanel.add(styleLabel);

        gbc.gridy = 1;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.gridx = 0;
        add(imagePanel,gbc);




        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.weightx = 1;
        gbc.weighty = 1;
        JPanel topPanel = new JPanel();
        add(topPanel, gbc);
        gbc.fill = GridBagConstraints.BOTH;


        gbc.weighty = 0;
        gbc.gridy = 3;
        gbc.anchor = GridBagConstraints.PAGE_END;

        // Create a panel with FlowLayout to place the buttons side by side.
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 0));

        FileSelect fs_content = new FileSelect();
        FileSelect fs_style = new FileSelect();


        JButton contentButton = new JButton("Select Content");
        //contentButton.addActionListener(fs_content);

        JButton styleButton = new JButton("Select Style");
        //styleButton.addActionListener(fs_style);

        JButton transferButton = new JButton("Transfer!");
        transferButton.setEnabled(false);

        // controller
        FileSelectSystem fs_system = new FileSelectSystem(contentButton,styleButton,transferButton,contentLabel,styleLabel);

        buttonPanel.add(contentButton);
        buttonPanel.add(styleButton);
        buttonPanel.add(transferButton);


        add(buttonPanel, gbc);

        setVisible(true);



    }
}