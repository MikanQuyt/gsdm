import os
import torch 
import yaml
import logging 
import numpy as np 
from PIL import Image
from torchvision import transforms
from model.model import GSDM
import util

path_image = r"./"
path_config = r"./config/final.yaml"

checkpoint_dir = r"./checkpoint/"
output_dir = r"./output/"

save_sp = False

def main():
    torch.manual_seed(1234)
    if torch.cuda.is_available(): #Kiem tra dem card roi co huu dung khong
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger = logging.getLogger("GSDM") # Hien thi ra man hinh
    logger.setLevel(logging.INFO) # De chi hien cap do INFO tro len
    handler = logging.StreamHandler() # Tao ra 1 luong de luu thong tin
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s")) # Định dạng log
    logger.addHandler(handler) # Them logger vao 

    with open(path_config, "r") as f:
        opt = yaml.load (f, Loader = yaml.FullLoader)

    opt["phase"] = "val" # Dat che do thanh val
    opt["SPM"]["resume_state"] = os.path.join(checkpoint_dir, "spm.pt") # Lay lai cau hinh model SPM da train
    opt["RM"]["path"]["resume_state"] = os.path.join(checkpoint_dir, "rm") # Lay lai cau hinh model RM da train

    gpu_list = ",".join(str(x) for x in opt["gpu_ids"]) # Lay danh sach cac GPU duoc su dung
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    opt["distributed"] = len(opt["gpu_ids"]) > 1 # Kiem tra xem co phai la he thong da GPU hay khong

    model = GSDM(opt)
    model.load_network(logger)
    logger.info("Model loaded successfully")


    img_path = os.path.abspath(path_image)
    assert os.path.isfile(img_path), f"Image not found: {img_path}"

    resolution = opt["val_dataset"]["resolution"] #[H, W]
    img_transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor = img_transform(img).unsqueeze(0)

    logger.info(f"Running inference on {img_path}")

    with torch.no_grad():
        output = model.inference(img_tensor)

    if save_sp:
        sp = util.gray2bgr(output["SPM"])
        final = torch.cat((sp, output["RM"]), dim = 2)
    else:
        final = output["RM"]

    sr_img = util.tensor2img(final)

    img_name = os.path.basename(img_path)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok = True)
        save_path = os.path.join(output_dir, img_name)
    else:
        name, ext = os.path.splitext(img_path)
        save_path = f"{name}_output{ext}"

    util.save_img(sr_img, save_path)
    logger.info(f"Output saved to: {save_path}")

if __name__ == "__main__":
    main()
    

