package org.app;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

public class FileSelect extends JFrame implements ActionListener {


    FileSelect() {

    }


    public void actionPerformed(ActionEvent evt){

        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        JFileChooser fs = new JFileChooser();

        //set directory to open
        fs.setCurrentDirectory(new File(System.getProperty("user.home")));
        int result = fs.showOpenDialog(null);

        if (result == JFileChooser.APPROVE_OPTION && fs.getSelectedFile().getName().contains(".jpeg")) {
            System.out.println(fs.getSelectedFile());
        }

    }
}


