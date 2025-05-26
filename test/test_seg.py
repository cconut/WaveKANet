import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt
from archs.archs import WaveKANet
from train.datasets import get_test_dataloaders
from train.metrics import iou_score
from train.utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  default='/root/autodl-tmp/result/models/CVC-ClinicDB-Split_WaveKANet/best_dice_model.pth',help='trained model path')
    parser.add_argument('--model_name', default='WaveKANet', help='model architecture')
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/data/CVC-ClinicDB-Split', help='dataset directory')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/CVC-ClinicDB-Split_results/WaveKANet', help='output directory')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')

    parser.add_argument('--image_size', default=256, type=int,
                        help='Input image dimension (default: 352)')
    return parser.parse_args()


def save_visualization(image, mask, pred, save_path, index):

    image = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC


    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')


    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')


    axes[2].imshow(pred.squeeze(), cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')


    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{index}.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()


def test(test_loader, model, output_dir):
    avg_meters = {
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'SE': AverageMeter(),
        'PC': AverageMeter(),
        'F1': AverageMeter(),
        'SP': AverageMeter(),
        'ACC': AverageMeter()
    }


    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    model.eval()


    sample_metrics = []
    sample_data = []

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for i, batch in enumerate(test_loader):
            input = batch['image'].cuda()
            target = batch['mask'].cuda()
            image_paths = batch['image_path']

            if args.deep_supervision:
                outputs = model(input)
                output = outputs[-1]
            else:
                output = model(input)

            iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)


            pred_masks = (output > 0.5).float()


            for b in range(input.size(0)):
                image_name = os.path.basename(image_paths[b])
                image_name = os.path.splitext(image_name)[0]
                sample_metrics.append({
                    'image_name': image_name,
                    'iou': iou,
                    'dice': dice,
                    'SE': SE,
                    'PC': PC,
                    'F1': F1,
                    'SP': SP,
                    'ACC': ACC
                })

                sample_data.append({
                    'image_name': image_name,
                    'image': input[b],
                    'mask': target[b],
                    'pred': pred_masks[b, 0],
                    'metrics': sample_metrics[-1]
                })

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            avg_meters['PC'].update(PC, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))
            avg_meters['SP'].update(SP, input.size(0))
            avg_meters['ACC'].update(ACC, input.size(0))

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('SE', avg_meters['SE'].avg),
                ('PC', avg_meters['PC'].avg),
                ('F1', avg_meters['F1'].avg),
                ('SP', avg_meters['SP'].avg),
                ('ACC', avg_meters['ACC'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()



    print("\nSaving all segmentation results...")
    for data in sample_data:

        save_visualization(
            data['image'],
            data['mask'],
            data['pred'],
            vis_dir,
            data['image_name']
        )


    sample_df = pd.DataFrame(sample_metrics)
    sample_df.to_csv(os.path.join(output_dir, 'sample_metrics.csv'), index=False)


    stats = {
        'metric': ['IoU', 'Dice', 'Sensitivity', 'Precision', 'F1', 'Specificity', 'Accuracy'],
        'mean': [
            avg_meters['iou'].avg,
            avg_meters['dice'].avg,
            avg_meters['SE'].avg,
            avg_meters['PC'].avg,
            avg_meters['F1'].avg,
            avg_meters['SP'].avg,
            avg_meters['ACC'].avg
        ],
        'std': [
            np.std(sample_df['iou']),
            np.std(sample_df['dice']),
            np.std(sample_df['SE']),
            np.std(sample_df['PC']),
            np.std(sample_df['F1']),
            np.std(sample_df['SP']),
            np.std(sample_df['ACC'])
        ]
    }

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)

    return OrderedDict([
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('SE', avg_meters['SE'].avg),
        ('PC', avg_meters['PC'].avg),
        ('F1', avg_meters['F1'].avg),
        ('SP', avg_meters['SP'].avg),
        ('ACC', avg_meters['ACC'].avg)
    ])



def main():
    global args
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    model = WaveKANet(num_classes=args.num_classes,img_size=args.image_size).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()

    test_loader = get_test_dataloaders(args.data_dir, batch_size=args.batch_size, img_size=(args.image_size,args.image_size))

    test_log = test(test_loader, model, args.output_dir)

    print('\nTest Results:')
    print('IoU: %.4f' % test_log['iou'])
    print('Dice: %.4f' % test_log['dice'])
    print('Sensitivity: %.4f' % test_log['SE'])
    print('Precision: %.4f' % test_log['PC'])
    print('F1: %.4f' % test_log['F1'])
    print('Specificity: %.4f' % test_log['SP'])
    print('Accuracy: %.4f' % test_log['ACC'])
    print(f'\nDetailed results have been saved to {args.output_dir}')
    print(f'Visualizations have been saved to {os.path.join(args.output_dir, "visualizations")}')


if __name__ == '__main__':
    main()

