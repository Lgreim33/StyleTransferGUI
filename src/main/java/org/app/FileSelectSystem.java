package org.app;

import ai.djl.ndarray.NDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class FileSelectSystem  {

    // Store buttons passed
    JButton contentButton;
    JButton styleButton;
    JButton transfer;

    JLabel contentImage;
    JLabel styleImage;

    FileSelect styleSelector;
    FileSelect contentSelector;

    // Constructor
    FileSelectSystem(JButton contentButton,JButton styleButton, JButton transferButton,JLabel contentLabel,JLabel styleLabel) {
        this.contentButton = contentButton;
        this.styleButton = styleButton;
        this.transfer = transferButton;

        this.contentImage = contentLabel;
        this.styleImage = styleLabel;

        this.styleSelector = new FileSelect();
        this.contentSelector = new FileSelect();

        // Add inline action listeners to the buttons
        this.contentButton.addActionListener(e -> onContentClick());
        this.styleButton.addActionListener(e -> onStyleClick());
        this.transfer.addActionListener(e -> onTransferClick());
    }


    void onContentClick(){
        this.contentSelector.openFileSelect();
        Image rawImage = new ImageIcon(contentSelector.selectedFile.toString()).getImage();
        Image scaledImage = rawImage.getScaledInstance(40,40,  java.awt.Image.SCALE_SMOOTH);
        this.contentImage.setIcon(new ImageIcon(scaledImage));
        bothButtonsClicked();
    }
    //
    void onStyleClick(){
        this.styleSelector.openFileSelect();
        Image rawImage = new ImageIcon(styleSelector.selectedFile.toString()).getImage();
        Image scaledImage = rawImage.getScaledInstance(40,40,  java.awt.Image.SCALE_SMOOTH);
        this.styleImage.setIcon(new ImageIcon(scaledImage));
        bothButtonsClicked();
    }

    // check to see if both buttons were clicked. If they were we can enable the transfer button
    void bothButtonsClicked(){
        if (this.styleSelector.fileSelected && this.contentSelector.fileSelected){
            this.transfer.setEnabled(true);

        }
    }

    // Upon clicking the transfer button, get the style and content images we selected earlier and commence with style transfer
    void onTransferClick(){
        Transfer transfer = new Transfer();
        if (transfer.model != null) {
            NDArray result = transfer.stylize(contentSelector.selectedFile, styleSelector.selectedFile);
        }

    }
}
