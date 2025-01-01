package org.app;

import ai.djl.Model;
import ai.djl.ndarray.*;
import ai.djl.inference.*;
import ai.djl.modality.cv.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.*;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Transfer {
    Model model = null;

    Transfer() {
        Path modelDir = Paths.get("src/Model/");


        try {
            this.model = Model.newInstance("model");

            try {
                this.model.load(modelDir);
            } catch (Exception ex) {
                System.out.println("Could not load model");
            }
        } catch (Exception ex) {
            System.out.println("Could not load model");
        }

    }


    NDArray stylize(File contentFile, File styleFile) {

        try (Predictor<NDList, NDList> predictor = this.model.newPredictor(new NoopTranslator())) {

            // Process the Style image
            Image styleImage = ImageFactory.getInstance().fromFile(Paths.get(styleFile.toString())).resize(512, 512, true);
            NDArray style = styleImage.toNDArray(this.model.getNDManager()).toType(DataType.FLOAT32, false).div(255f);
            style = style.expandDims(0);
            style = style.transpose(0, 3, 1, 2);

            // Process the content image
            Image contentImage = ImageFactory.getInstance().fromFile(Paths.get(contentFile.toString())).resize(512, 512, true);
            NDArray content = contentImage.toNDArray(model.getNDManager()).toType(DataType.FLOAT32, false).div(255f);
            content = content.expandDims(0);
            content = content.transpose(0, 3, 1, 2);

            // Perform inference
            NDList output = predictor.predict(new NDList(content, style));


            NDArray result = output.head();
            result = result.squeeze(0);

            result = result.mul(255f).clip(0, 255).toType(DataType.UINT8, false);


            // If your data is in [C, H, W], you can directly convert it:
            Image outputImage = ImageFactory.getInstance().fromNDArray(result);

            // If needed, you can save the image to a file:
            outputImage.save(Files.newOutputStream(Paths.get("output.jpg")), "jpg");

            return result;

        } catch (Exception e) {
            System.out.println("Could not create new predictor");
            e.printStackTrace();

        }

        return null;

    }
}