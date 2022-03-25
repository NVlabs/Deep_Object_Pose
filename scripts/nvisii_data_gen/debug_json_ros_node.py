import json
import time

import rospy
import tf
from tf.transformations import quaternion_from_matrix, translation_from_matrix
import numpy as np

rospy.init_node("debug_json")

tf_broadcaster = tf.TransformBroadcaster()

while True:
    for frame_number in range(50):
        if rospy.is_shutdown():
            break
        path = f"{str(frame_number).zfill(5)}.json"
        with open(path) as json_file:
            conf = json.load(json_file)
        print(path)

        stamp = rospy.Time.now()

        camera_data = conf['camera_data']
        tf_broadcaster.sendTransform(translation=camera_data['location_worldframe'],
                                     rotation=camera_data['quaternion_xyzw_worldframe'],
                                     time=stamp,
                                     parent='world',
                                     child='camera',
                                     )
        # transpose to transform between column-major and row-major order
        camera_view_matrix = np.array(camera_data['camera_view_matrix']).transpose()
        tf_broadcaster.sendTransform(translation=translation_from_matrix(camera_view_matrix),
                                     rotation=quaternion_from_matrix(camera_view_matrix),
                                     time=stamp,
                                     parent='camera',
                                     child='world_from_matrix',
                                     )
        for object_data in conf['objects']:
            tf_broadcaster.sendTransform(translation=object_data['location_worldframe'],
                                         rotation=object_data['quaternion_xyzw_worldframe'],
                                         time=stamp,
                                         parent='world',
                                         child=f"{object_data['name']}_world",
                                         )
            tf_broadcaster.sendTransform(translation=object_data['location'],
                                         rotation=object_data['quaternion_xyzw'],
                                         time=stamp,
                                         parent='camera',
                                         child=f"{object_data['name']}_cam",
                                         )
            local_to_world_matrix = np.array(object_data['local_to_world_matrix']).transpose()
            tf_broadcaster.sendTransform(translation=translation_from_matrix(local_to_world_matrix),
                                         rotation=quaternion_from_matrix(local_to_world_matrix),
                                         time=stamp,
                                         parent='world',
                                         child=f"{object_data['name']}_cam_from_matrix",
                                         )

        time.sleep(1 / 30)
