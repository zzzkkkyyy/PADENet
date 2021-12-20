import numpy as np
import math
import os
import sys
import argparse

delta = 1.25
num_images = 200

def compute_metrics(prediction, groundtruth, mask):
    prediction = prediction[mask]
    groundtruth = groundtruth[mask]

    diff = prediction - groundtruth

    abs_rel = np.mean(np.abs(diff) / groundtruth)
    sq_rel = np.mean(diff**2 / groundtruth)
    rmse = np.sqrt(np.mean(np.abs(diff)))
    rmse_log = np.sqrt(np.mean(np.abs(np.log(np.clip(prediction, 1e-2, 1e4)) - np.log(groundtruth))))

    ratio1 = prediction / groundtruth
    ratio2 = groundtruth / prediction

    max_ratio = np.maximum(ratio1, ratio2)
    threshold = np.mean(np.less(max_ratio, delta))
    threshold2 = np.mean(np.less(max_ratio, delta ** 2))
    threshold3 = np.mean(np.less(max_ratio, delta ** 3))

    return (abs_rel, sq_rel, rmse, rmse_log, threshold, threshold2, threshold3)

def evaluate(gt_dir, image_set_file, result_dir, pred_as_disparity):

    image_set = open(image_set_file).readlines()

    total_metrics = []
    for name in sorted(image_set)[:num_images]:
        name = name.strip()+".npy"

        # Load the groundtruth data
        groundtruth_filename = os.path.join(gt_dir, name)
        groundtruth_meters = np.load(groundtruth_filename, allow_pickle=False)
        width = groundtruth_meters.shape[1]
        height = groundtruth_meters.shape[0]

        # Ignore pixels for which the groundtruth depth is either unknown or greater than 50m.
        mask = np.logical_and(np.greater(groundtruth_meters, 0.0), np.less(groundtruth_meters, 50))

        # We evaluate using the disparity rather than the distance.
        angular_precision = width/math.pi
        groundtruth = angular_precision / groundtruth_meters

        # Load the prediction
        prediction_filename = os.path.join(result_dir, name)
        prediction = np.load(prediction_filename, allow_pickle=False)

        if not pred_as_disparity:
            # If the prediction is in meters, convert to disparity.
            prediction = angular_precision / prediction

        # Compute all metrics.
        metrics = compute_metrics(prediction, groundtruth, mask)
        total_metrics.append(metrics)

    # Compute the average of all the metrics across each image.
    total_metrics = list(zip(*total_metrics))
    total_metrics = [np.mean(m) for m in total_metrics]

    # Print the results
    print("Abs. rel. & Sq. rel. & RMSE & RMSE log. & Depth acc < %s    %s    %s" % (str(delta), str(delta ** 2), str(delta ** 3)))
    print(" & ".join(["%.3f" % m for m in total_metrics]))

def main():
    parser = argparse.ArgumentParser(description=
            """Depth estimation evaluation script.

            Unless --pred-as-disparity is set, the predictions are expected to be in meters.
            If --pred-as-disparity, the predictions values are expected to equal to angular_precision/depth_in_meter
            where angular_precision=width/pi.
            """)
    parser.add_argument('--gt-dir', type=str, default='/media/zhoukeyang/软件/synthetic-panoramic-dataset/val/depth',
            help="Groundtruth labels directory")
    parser.add_argument('--set', type=str, default='/media/zhoukeyang/软件/synthetic-panoramic-dataset/val/ImageSets/val.txt',
            help="Image set file")
    parser.add_argument('--result-dir', required=True, type=str,
            help="Results directory")
    parser.add_argument('--pred-as-disparity', action='store_false',
            help="Prediction as disparity values (instead of distance in meters)")

    args = parser.parse_args()

    evaluate(args.gt_dir, args.set, args.result_dir, args.pred_as_disparity)

if __name__ == '__main__':
    main()
