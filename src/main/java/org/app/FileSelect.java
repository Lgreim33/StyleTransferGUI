package org.app;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
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
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Image Files", "jpg", "png", "gif", "jpeg");
        JFileChooser fs = new JFileChooser();
        fs.setFileFilter(filter);

        //set directory to open
        fs.setCurrentDirectory(new File(System.getProperty("user.home")));
        int result = fs.showOpenDialog(null);

        if (result == JFileChooser.APPROVE_OPTION)
        {
            System.out.println(fs.getSelectedFile());
        }

    }
}


