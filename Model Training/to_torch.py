import model
import torch



# retreive model
transferModel = model.StyleTransferModel()

transferModel.load_state_dict(torch.load("ssim_styleTran.pth", map_location="cpu")["model_state_dict"])
transferModel.eval()


# Script and save model
scripted_model = torch.jit.script(transferModel)
scripted_model.save("model.pt")