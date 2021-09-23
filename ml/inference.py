
import torch
import numpy as np
import os
import cv2
import argparse
import utils
from importlib import import_module

def generate_grad_cam(args, configure):

    # -- find and load best model (highest acc)
    model_dir = os.path.join(configure["config"]["save_dir"])

    model_files = utils.get_model_files(model_dir)
    best_model = model_files[utils.get_best_model_index(model_files)]

    model = torch.load(os.path.join(model_dir, best_model))

    # -- augmentation
    transform_module = getattr(import_module("dataset"), configure["config"]["hparam"]["augmentation"])
    transform = transform_module(
        resize = configure["config"]["hparam"]["resize"],
    )
    # -- dataset
    dataset_module = getattr(import_module("dataset"), configure["config"]["dataset"])
    dataset = dataset_module(
        root = configure["config"]["data_dir"],
        transform=transform,
    )

    # -- data_loader
    val_set = dataset.test_data
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.sample,
        drop_last=True,
        shuffle=False,
    )

    # -- json
    result = {
        "grad_cam":{
            "blending":[],
            "heatmap":[],
            "original":[],
        }
    }
    # -- load sample image

    sample = next(iter(val_loader))
    sample = sample[0]

    model.eval()
    gradients = model.get_activations_gradient()

    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
    for idx, (img) in enumerate(sample):
        img = img.cuda()
        activations = model.get_activations(img.unsqueeze(0)).detach()
        
        for i in range(img.size(1)):
            activations[:,i,:,:] += pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze().cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)

        img = img.cpu().permute(1,2,0).numpy()

        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        
        img = np.uint8(255 * img)
        superimposed_img = (heatmap) * 0.4 + img

        img = cv2.resize(img, args.output_size)
        heatmap = cv2.resize(heatmap, args.output_size)
        superimposed_img = cv2.resize(superimposed_img, args.output_size)

        heatmap_path = os.path.join(model_dir, f'heatmap_{idx}.jpg')
        original_path = os.path.join(model_dir, f'original_{idx}.jpg')
        superimposed_img_path = os.path.join(model_dir, f'blending_{idx}.jpg')

        result["grad_cam"]["blending"].append(superimposed_img_path)
        result["grad_cam"]["heatmap"].append(heatmap_path)
        result["grad_cam"]["original"].append(original_path)

        cv2.imwrite(heatmap_path, heatmap)
        cv2.imwrite(original_path, img)
        cv2.imwrite(superimposed_img_path, superimposed_img)
    
    utils.save_inference_result(model_dir, result)
    

if __name__ =="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='a path of model for inference')
    parser.add_argument('--sample', type=int, default=3, help='count of sample image')
    parser.add_argument('--output_size', type=int, nargs="+", default=[224,224], help='visualization output size (default: 32,32)')
    args = parser.parse_args()
    print(args)

    configure = utils.get_json(os.path.join(args.path, "configure.json"))
    generate_grad_cam(args, configure)
