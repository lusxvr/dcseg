import argparse
import os

from data_processing.scannet.sensor_data import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--filename', required=True, help='path to sens file to read')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print(opt)


def main():
  if not os.path.exists(opt.output_path):
    os.makedirs(opt.output_path)

  sd = SensorData(opt.filename)

  sd.export_depth_images(os.path.join(opt.output_path, 'depth'))
  sd.export_color_images(os.path.join(opt.output_path, 'color'))
  sd.export_poses(os.path.join(opt.output_path, 'pose'))
  sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))


if __name__ == '__main__':
    main()