import argparse
import os
import torch
import cv2


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-image', type=str,
                        required=True, help='The Input Image.')

    parser.add_argument('-m', '--model', type=str,
                        default='./traffic_signs_best_46_epo_94_acc.pt',
                        required=False, help="The Trained YOLOv5 Model.")

    parser.add_argument('-d', '--device', type=str, default='CPU', required=False,
                        help="The Inference Device Whether: 'CPU', 'CUDA:0', 'CUDA:1', etc...")

    parser.add_argument('-s', '--size', type=int, default=640,
                        required=False, help="Images Inference Size.")

    args = parser.parse_args()
    input_image = args.input_image
    assert os.path.exists(input_image), "Image Path Doesn't Exist."

    model = args.model
    assert os.path.exists(model), "Model Path Doesn't Exist."

    return input_image, model, args.size, args.device


def run(image,
        model_path='./traffic_signs_best_46_epo_94_acc.pt',
        device='cpu',
        size=1280,
        ):
    """
    image: can be: file, Path, PIL, OpenCV, numpy, list
    save_img: wheter to save the result image or not (default False)
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=model_path,
                           )
    if 'cuda' in device:
        model.cuda()

    data = model(image, size=size)
    preds = data.pred[0]
    data.display(render=True)
    cv2_img = cv2.cvtColor(data.imgs[0], cv2.COLOR_RGB2BGR)

    path, ext = os.path.splitext(image)
    save_path = path+'_output_'+ext
    print('save_path: ', f'"{save_path}"')
    cv2.imwrite(save_path, cv2_img)


if __name__ == '__main__':
    input_image, model, size, device = init_args()
    run(input_image,  # './training_data/images/val/00002.jpg',
        model_path=model,
        device=device,
        size=size)
