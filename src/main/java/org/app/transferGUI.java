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
        setTitle("My GUI");
        setSize(400,400);

        setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.anchor = GridBagConstraints.CENTER;

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