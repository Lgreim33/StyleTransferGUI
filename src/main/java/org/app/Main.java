package org.app;

import ai.djl.Model;
import ai.djl.ndarray.*;

import java.nio.file.Files;
import java.nio.file.Path;

import ai.djl.inference.*;
import ai.djl.modality.cv.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.*;


import java.nio.file.Paths;


public class Main {
    public static void main(String[] args) {

        Transfer transferMod = new Transfer();
        NDArray result = transferMod.stylize();


        transferGUI gui = new transferGUI();


    }
}




