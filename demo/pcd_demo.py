import os
from argparse import ArgumentParser
from tqdm import tqdm

from mmdet3d.apis import inference_detector, init_detector, show_result_meshlab


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='/home/ssm-user/xiaorui/lidar/qualcomm/pred', help='dir to save results')
    args = parser.parse_args()
    checkpoint_name = os.path.basename(args.checkpoint).split('.')[0]

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    out_dir = os.path.join(args.out_dir, checkpoint_name)
    # test a single image
    if args.pcd.endswith(".bin"):
        result, data = inference_detector(model, args.pcd)
        # show the results
        show_result_meshlab(data, result, out_dir, args.score_thr)
    elif args.pcd.endswith(".txt"):
        point_bins = [l.strip() for l in open(args.pcd, "r").readlines()]
        for point_bin in tqdm(point_bins):
            result, data = inference_detector(model, point_bin)
            # show the results
            show_result_meshlab(data, result, out_dir , args.score_thr)
    else:
        print("Invalid pcd")

if __name__ == '__main__':
    main()
