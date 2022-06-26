# Debugging Tool to Visualize Synthetic Data Projected Points Accuracy


from PIL import ImageDraw, Image
import numpy as np 
import json


class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(0, 0, 255)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)

path_json = "/home/andrewg/Documents/dope_ws/generated_data/5000_DOME_Cracker_Box/data/000009.json"
path_img = "/home/andrewg/Documents/dope_ws/generated_data/5000_DOME_Cracker_Box/data/000009.png"
path_draw = "/home/andrewg/Documents/dope_ws/src/dope/scripts/train2/output/inference/debug_draw/draw.png"

img = Image.open(path_img).convert('RGB')

with open(path_json) as f:
    data_json = json.load(f)

draw = Draw(img)

for obj in data_json['objects']:
    projected_cuboid_keypoints = [tuple(pair) for pair in obj['projected_cuboid']]
    draw.draw_cube(projected_cuboid_keypoints)

img.save(path_draw)
