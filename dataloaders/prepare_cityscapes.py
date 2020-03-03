#
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
    prepare_cityscapes.py
    ~~~~~~~~~~~~~~~~~~~~~

    Prepare the Cityscapes dataset from a folder with all the sequences.
"""
import os
import argparse
import cv2
from glob import glob
import parse
import pandas as pd


FNAME_PARSER = parse.compile('{city}_{seq_id:d}_{frame_id:d}_leftImg8bit.png')


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, required=True,
                        help='Directory with images to resize')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to output the resized images')
    parser.add_argument('--img_side', type=int, required=True,
                        help='Side of the resized images.')

    args = parser.parse_args()

    return args


def resize_image(in_file, out_file, img_side):
    img = cv2.imread(in_file)
    h, w = img.shape[:2]
    w_margin = 512 
    h_margin = 0
    # img = img[h_margin:-h_margin, w_margin:-w_margin]
    img = img[:, w_margin:-w_margin]
    img = cv2.resize(img, (img_side, img_side), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_file, img, [cv2.IMWRITE_JPEG_QUALITY, 98])


def main(in_dir, out_dir, img_side):

    # List cities in the input directory
    cities = os.listdir(in_dir)

    # List directories in the input dir
    all_fnames = []
    for city in cities:

        in_city_dir = os.path.join(in_dir, city)

        # List frames
        frame_fnames = os.listdir(in_city_dir)
        all_fnames = all_fnames + [FNAME_PARSER.parse(f).named for f in frame_fnames]

    # Convert into a pandas dataframe
    df = pd.DataFrame(all_fnames)

    # Loop over cities
    for city in cities:

        # Define dirs
        in_city_dir = os.path.join(in_dir, city)
        out_city_dir = os.path.join(out_dir, city)

        # Find all sequences for that city
        city_df = df[df.city == city]
        city_seq_ids = city_df.seq_id.unique()

        for seq_id in city_seq_ids:

            out_seq_dir = os.path.join(out_city_dir, '{:0>6}'.format(seq_id))

            # List all the fnames
            seq_df = city_df[city_df.seq_id == seq_id]

            # Resize all the images
            if len(seq_df) == 30:

                if not os.path.exists(out_seq_dir):
                    os.makedirs(out_seq_dir)

                for index, row in seq_df.iterrows():

                    in_file = os.path.join(in_city_dir, '{}_{:0>6}_{:0>6}_leftImg8bit.png'.format(city, seq_id, row['frame_id']))
                    out_file = os.path.join(out_seq_dir, '{:0>6}.jpg'.format(row['frame_id']))
                    resize_image(in_file, out_file, img_side)

                print('{}/{} done'.format(city, seq_id))

            else:
                print('{}/{} skipped'.format(city, seq_id))


    print('All done')


if __name__ == '__main__':
    args = parse_args()
    main(args.in_dir, args.out_dir, args.img_side)
