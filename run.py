# obtain MAX DS size
# get MIN DS size
# create augmented images for MIN DS


import os

import Augmentor

image_path = os.path.expanduser('~/Documents/images/dataset/visage_v1.1b/12/')
output_directory = os.path.expanduser('~/Documents/images/dataset/augmented/')

probability = 1

p = Augmentor.Pipeline(
    source_directory=image_path,
    output_directory=output_directory
)
p.flip_left_right(probability=probability)
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.zoom_random(probability=probability, percentage_area=0.95)
p.random_distortion(probability=probability, grid_width=2, grid_height=2, magnitude=8)
p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
p.random_erasing(probability=probability, rectangle_area=0.2)


for i in range(0, 2):
    print(i)
    p.process()

print('processed')