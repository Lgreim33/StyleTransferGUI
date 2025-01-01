package org.app;

import javax.swing.*;
import javax.swing.plaf.nimbus.NimbusLookAndFeel;
import java.awt.*;

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
        setSize(400,400);



        setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();


        gbc.gridx = 0;
        gbc.gridy = 0;
        //gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        JLabel appLabel = new JLabel("Style Transfer Demo Application");
        add(appLabel, gbc);

        // Components that will display the selected style and content images
        JLabel contentIcon = new JLabel();
        JLabel styleIcon = new JLabel();

        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.weightx = 1;
        gbc.weighty = 1;
        JPanel topPanel = new JPanel();
        add(topPanel, gbc);
        gbc.fill = GridBagConstraints.BOTH;


        gbc.weighty = 0;
        gbc.gridy = 2;
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
        FileSelectSystem fs_system = new FileSelectSystem(contentButton,styleButton,transferButton);

        buttonPanel.add(contentButton);
        buttonPanel.add(styleButton);
        buttonPanel.add(transferButton);


        add(buttonPanel, gbc);

        setVisible(true);



    }
}