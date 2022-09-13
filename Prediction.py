import os
import argparse
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.network.faster_rcnn_framework import FasterRCNN
from utils.network import boxes as box_ops
from utils.network.resnet50_fpn_model import resnet50_fpn_backbone
from utils.draw_box_utils import draw_box
from utils.draw_box_utils import box_filter_based_scores


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    assert os.path.exists(args.input_path), f"The {args.input_path} does not exist."
    image_files = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path)
                   if os.path.splitext(file)[1] == '.jpg']
    assert len(image_files) > 0, f"There is no image in {args.input_path}."
    image_draw = os.path.join(args.input_path, args.image_draw)
    assert os.path.exists(image_draw), "The image_draw file does not exist."
    assert args.image_draw in os.listdir(args.input_path), "The image_draw is not in image list."
    image_draw = Image.open(image_draw)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = create_model(num_classes=2)

    weights_file = r'./weights/weights_file.pth'
    assert os.path.exists(weights_file), "{} file dose not exist.".format(weights_file)
    model.load_state_dict(torch.load(weights_file, map_location=device)["model"])
    model.to(device)

    class_dict = {"defect": 1}
    category_index = {v: k for k, v in class_dict.items()}

    results_file = os.path.join(args.output_path,
                                "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    with open(results_file, 'w') as logfile:
        logfile.write('The prediction process:\n')

    all_boxes = []
    all_scores = []
    all_classes = []
    model.eval()
    t_start = time_synchronized()
    for num, image in enumerate(image_files):
        original_img = Image.open(image).convert('RGB')
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t1 = time_synchronized()
            predictions = model(img.to(device))[0]
            t2 = time_synchronized()
            print(f"[INFO] Inference time of image {num}: {(t2 - t1): .6f}s")
            with open(results_file, 'a') as logfile:
                logfile.write(f"Inference time of image {num}: {(t2 - t1): .6f}s\n")

            predict_boxes = predictions["boxes"]
            predict_classes = predictions["labels"]
            predict_scores = predictions["scores"]

            if len(predict_boxes) == 0:
                print("No defect is detected.")
                continue
            else:
                all_boxes.append(predict_boxes)
                all_classes.append(predict_classes)
                all_scores.append(predict_scores)
    if len(all_boxes) == 0:
        print("No defect is detected in the images. The program will be stopped.")
        exit()
    all_boxes, all_classes, all_scores = box_filter_based_scores(all_boxes, all_classes, all_scores,
                                                                 args.scores_threshold)
    all_boxes = torch.stack(all_boxes)
    all_classes = torch.stack(all_classes)
    all_scores = torch.stack(all_scores)
    keep = box_ops.batched_nms(all_boxes, all_scores, all_classes, args.nms_thresh)
    all_boxes, all_scores, all_classes = all_boxes[keep], all_scores[keep], all_classes[keep]

    t_end = time_synchronized()
    print(f"[INFO] The total inference time: {(t_end - t_start): .6f}s")
    with open(results_file, 'a') as logfile:
        logfile.write(f"The total inference time: {(t_end - t_start): .6f}s\n")

    draw_box(image_draw,
             all_boxes.cpu().numpy(),
             all_classes.cpu().numpy(),
             all_scores.cpu().numpy(),
             category_index,
             thresh=0.5,
             line_thickness=8)
    plt.imshow(image_draw)
    plt.show()
    # Save the prediction result
    output_file = os.path.join(args.output_path, 'prediction.jpg')
    image_draw.save(output_file, dpi=(300.0, 300.0))
    print("[INFO] The program is finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input_path', default=r'./example case', help='input path')
    parser.add_argument('--output_path', default=os.path.join(parser.parse_args().input_path, 'prediction'),
                        help='output path')
    parser.add_argument('--scores_threshold', default=0.989, help='The threshold to filter the bounding box.')
    parser.add_argument('--nms_thresh', default=0.025, help='The NMS threshold.')
    parser.add_argument('--image_draw', default='FBHs_PCA_1.jpg',
                        help='The name of the image used to draw the prediction results.')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    config_file = os.path.join(args.output_path, 'config.txt')
    with open(config_file, 'w') as f:
        for var in str(args).split(','):
            f.write(var+',\n')
    main(args)
