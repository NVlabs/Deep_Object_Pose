#!/usr/bin/env python3

import json
import numpy as np
import os
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import sys


def main(json_files):
    for json_fn in json_files:
        # Find corresponding PNG
        base, _ = os.path.splitext(json_fn)
        img_fn = base+'.png'
        if not os.path.isfile(img_fn):
            print(f"Could not locate '{img_fn}'. Skipping..")
            continue

        # Load JSON data
        with open(json_fn, 'r') as F:
            data_json = json.load(F)
        up = np.array(data_json['camera_data']['camera_look_at']['up'])
        at = np.array(data_json['camera_data']['camera_look_at']['at'])
        eye = np.array(data_json['camera_data']['camera_look_at']['eye'])

        cam_matrix = np.eye(4)
        cam_matrix[0:3,0] = up
        cam_matrix[0:3,1] = np.cross(up, -at)
        cam_matrix[0:3,2] = -at
        cam_matrix[0:3,3] = -eye

        img = Image.open(img_fn)

        objects = data_json['objects']
        # draw projected cuboid dots
        for oo in objects:
            draw = ImageDraw.Draw(img)
            pts = oo['projected_cuboid']
            for pt in pts:
                draw.ellipse((pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2), fill = 'cyan',
                             outline ='cyan')

            line_order = [[0, 1], [1, 2], [3, 2], [3, 0], # front
                          [4, 5], [6, 5], [6, 7], [4, 7], # back
                          [0, 4], [7, 3], [5, 1], [2, 6], # sides
                          [0, 5], [1,4]]                  # 'x' on top
            for ll in line_order:
                draw.line([(pts[ll[0]][0],pts[ll[0]][1]), (pts[ll[1]][0],pts[ll[1]][1])],
                           fill='cyan', width=1)

        img.save(base+'-output.png')


def usage_msg(script_name):
    print(f"Usage: {script_name} _JSON FILES_")
    print("  The basename of the JSON files in _FILES_ will be used to find its")
    print("  corresponding image file; i.e. if `00001.json` is provided, the code")
    print("  will look for an image named `00001.png`")


if __name__ == "__main__":
    # Print out usage information if there are no arguments
    if len(sys.argv) < 2:
        usage_msg(sys.argv[0])
        exit(0)

    # ..or if the first argument is a request for help
    s = sys.argv[1].lstrip('-')
    if s == "h" or s == "help":
        usage_msg(sys.argv[0])
        exit(0)

    main(sys.argv[1:])


