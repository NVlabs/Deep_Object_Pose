"""
This file starts a ROS2 node to run DOPE, 
listening to an image topic and publishing poses.
"""

# Python packages
import cv2
import message_filters
import numpy as np
import resource_retriever
from PIL import Image
from PIL import ImageDraw
from cv_bridge import CvBridge
# from nptyping import NDArray

import transformations

# Custom library
from inference_script.cuboid import Cuboid3d
from inference_script.cuboid_pnp_solver import CuboidPNPSolver
from inference_script.detector import ModelData, ObjectDetector
import simple_colors
# ROS2 packages
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image as ImageSensor_msg
from std_msgs.msg import String
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


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

    def draw_cube(self, points, color=(255, 0, 0)):
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


class DopeNode(Node):
    def __init__(self):
        super().__init__('dope_node', 
                         automatically_declare_parameters_from_overrides=True, allow_undeclared_parameters=True)
        
        print(simple_colors.cyan('Setting parameters for DOPE...',['bold']))

        self.obj_detector = ObjectDetector()

        self.pubs = {}
        self.models = {}
        self.pnp_solvers = {}
        self.pub_dimension = {}
        self.pub_belief = {}
        self.draw_colors = {}
        self.dimensions = {}
        self.class_ids = {}
        self.model_transforms = {}
        self.meshes = {}
        self.mesh_scales = {}
        self.cv_bridge = CvBridge()
        self.prev_num_detections = 0

        self.input_is_rectified = self.get_parameter_or(
            'input_is_rectified', True).get_parameter_value().bool_value
        self.downscale_height = self.get_parameter_or('downscale_height', 500).get_parameter_value().integer_value
        self.overlay_belief_images = self.get_parameter_or(
            'overlay_belief_images', True).get_parameter_value().bool_value

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = self.get_parameter_or(
            'thresh_angle', 0.5).get_parameter_value().double_value
        self.config_detect.thresh_map = self.get_parameter_or(
            'thresh_map', 0.01).get_parameter_value().double_value
        self.config_detect.sigma = self.get_parameter_or('sigma', 3).get_parameter_value().integer_value
        self.config_detect.thresh_points = self.get_parameter_or(
            "thresh_points", 0.1).get_parameter_value().double_value
        

        # For each object to detect, load network model, create PNP solver, and start ROS publishers
        for model, weights_url in self.get_parameters_by_prefix('weights').items():
            # print(model, weights_url.get_parameter_value().string_value)
            
            self.models[model] = \
                ModelData(
                    model,
                    resource_retriever.get_filename(
                        weights_url.get_parameter_value().string_value, use_protocol=False)
            )
            self.models[model].load_net_model()

            try:
                M = self.get_parameters_by_prefix('model_transforms')[model].get_parameter_value().double_array_value
                # self.model_transforms[model] = tf.transformations.quaternion_from_matrix(M)
                self.model_transforms[model] = transformations.quaternion_from_matrix(M) #utils_py.quaternion_from_matrix(M)
            except KeyError:
                self.model_transforms[model] = np.array([1.0, 0.0, 0.0, 0.0], dtype='float64')

            try:
                self.meshes[model] = self.get_parameters_by_prefix('meshes')[model].get_parameter_value().string_value
            except KeyError:
                pass


            try:
                self.mesh_scales[model] = self.get_parameters_by_prefix('mesh_scales')[model].get_parameter_value().double_value
            except KeyError:
                self.mesh_scales[model] = 1.0


            try:
                self.draw_colors[model] = tuple(self.get_parameters_by_prefix('draw_colors')[model].get_parameter_value().integer_array_value)
            except:
                self.draw_colors[model] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

            self.dimensions[model] = tuple(
                self.get_parameters_by_prefix('dimensions')[model].get_parameter_value().double_array_value)
            self.class_ids[model] = self.get_parameters_by_prefix('class_ids')[model].get_parameter_value().integer_value

            self.pnp_solvers[model] = \
                CuboidPNPSolver(
                    model,
                    cuboid3d=Cuboid3d(self.get_parameters_by_prefix('dimensions')[model].get_parameter_value().double_array_value)
            )

            self.pubs[model] = \
                self.create_publisher(
                    PoseStamped,
                    '{}/pose_{}'.format(self.get_parameter('topic_publishing').get_parameter_value().string_value, model),
                    10
            )

            self.pub_dimension[model] = \
                self.create_publisher(
                    String,
                    '{}/dimension_{}'.format(
                        self.get_parameter('topic_publishing').get_parameter_value().string_value, model),
                    10
            )

            self.pub_belief[model] = \
                self.create_publisher(
                    ImageSensor_msg,
                    '{}/belief_{}'.format(self.get_parameter('topic_publishing').get_parameter_value().string_value, model),
                    10
            )
            
            print(simple_colors.cyan("Publishers successfully created for " + model + " model.", ['bold']))

        # # Start ROS publishers
        self.pub_rgb_dope_points = \
            self.create_publisher(
                ImageSensor_msg,
                self.get_parameter('topic_publishing').get_parameter_value().string_value + "/rgb_points",
                10
            )

        self.pub_camera_info = \
            self.create_publisher(
                CameraInfo,
                self.get_parameter('topic_publishing').get_parameter_value().string_value + "/camera_info",
                10
            )

        self.pub_detections = \
            self.create_publisher(
                Detection3DArray,
                '~/detected_objects',
                10
            )

        self.pub_markers = \
            self.create_publisher(
                MarkerArray,
                '~/markers',
                10
            )

        # # # # Start ROS subscriber
        image_sub = message_filters.Subscriber(self, ImageSensor_msg,
            self.get_parameter('topic_camera').get_parameter_value().string_value,
            qos_profile=rclpy.qos.qos_profile_sensor_data
        )

        info_sub = message_filters.Subscriber(self, CameraInfo,
            self.get_parameter('topic_camera_info').get_parameter_value().string_value,
            qos_profile=rclpy.qos.qos_profile_sensor_data
        )

        ts = message_filters.TimeSynchronizer([image_sub, info_sub], 1)
        ts.registerCallback(self.image_callback)
        print(simple_colors.cyan("Subscriber successfully created.", ['bold']))

        print(simple_colors.cyan("Running DOPE...", [
              'bold']) + "\nListening to camera topic: '{}'".format(self.get_parameter('topic_camera').get_parameter_value().string_value))
        print("\nCtrl-C to stop")


    def image_callback(self, image_msg, camera_info):
        
        """Image callback"""
        msg_dim = String()
        img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        # cv2.imwrite('img.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # for debugging

        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            # P: NDArray[(3, 4), float] = np.matrix(camera_info.p)
            P = np.matrix(camera_info.p, dtype='float64')
            P.resize((3, 4))
            camera_matrix = P[:, :3]

            dist_coeffs = np.zeros((4, 1))
        else:
            camera_matrix = np.matrix(camera_info.k, dtype='float64')
            camera_matrix.resize((3, 3))

            dist_coeffs = np.matrix(camera_info.d, dtype='float64')
            dist_coeffs.resize((len(camera_info.d), 1))

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(img, (int(scaling_factor * width),
                             int(scaling_factor * height)))

        for m in self.models:
            self.pnp_solvers[m].set_camera_intrinsic_matrix(camera_matrix)
            self.pnp_solvers[m].set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        detection_array = Detection3DArray()
        detection_array.header = image_msg.header


        for m in self.models:
            publish_belief_img = (self.pub_belief[m].get_subscription_count() > 0)

            # Detect object
            results, im_belief = self.obj_detector.detect_object_in_image(
                self.models[m].net,
                self.pnp_solvers[m],
                img,
                self.config_detect,
                make_belief_debug_img=publish_belief_img,
                overlay_image=self.overlay_belief_images
            )


            # Publish pose and overlay cube on image
            for i_r, result in enumerate(results):
                
                if result["location"] is None:
                    continue
                loc = result["location"]
                ori_ = result["quaternion"]
                ori = np.array([ori_[3], ori_[0], ori_[1], ori_[2]], dtype='float64')

                transformed_ori = transformations.quaternion_multiply(ori, self.model_transforms[m])
                

                # rotate bbox dimensions if necessary
                # (this only works properly if model_transform is in 90 degree angles)
                dims = rotate_vector(
                    vector=self.dimensions[m], quaternion=self.model_transforms[m])
                dims = np.absolute(dims)
                dims = tuple(dims)

                pose_msg = PoseStamped()
                pose_msg.header = image_msg.header
                CONVERT_SCALE_CM_TO_METERS = 100
                pose_msg.pose.position.x = loc[0] / CONVERT_SCALE_CM_TO_METERS
                pose_msg.pose.position.y = loc[1] / CONVERT_SCALE_CM_TO_METERS
                pose_msg.pose.position.z = loc[2] / CONVERT_SCALE_CM_TO_METERS
                pose_msg.pose.orientation.x = transformed_ori[1]
                pose_msg.pose.orientation.y = transformed_ori[2]
                pose_msg.pose.orientation.z = transformed_ori[3]
                pose_msg.pose.orientation.w = transformed_ori[0]

                # Publish
                self.pubs[m].publish(pose_msg)
                
                msg_dim.data = str(dims)
                self.pub_dimension[m].publish(msg_dim)
                # self.pub_dimension[m].publish(String(dims))

                # Add to Detection3DArray
                detection = Detection3D()
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(self.class_ids[result["name"]])
                hypothesis.hypothesis.score = float(result["score"])
                hypothesis.pose.pose = pose_msg.pose
                detection.results.append(hypothesis)
                detection.bbox.center = pose_msg.pose
                detection.bbox.size.x = dims[0] / CONVERT_SCALE_CM_TO_METERS
                detection.bbox.size.y = dims[1] / CONVERT_SCALE_CM_TO_METERS
                detection.bbox.size.z = dims[2] / CONVERT_SCALE_CM_TO_METERS
                detection_array.detections.append(detection)

                # Draw the cube
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))
                    draw.draw_cube(points2d, self.draw_colors[m])



            # Publish the belief image
            if publish_belief_img:
                belief_img = self.cv_bridge.cv2_to_imgmsg(
                    np.array(im_belief)[..., ::-1], "bgr8")
                belief_img.header = camera_info.header
                self.pub_belief[m].publish(belief_img)


        # Publish the image with results overlaid
        
        rgb_points_img = CvBridge().cv2_to_imgmsg(
            np.array(im)[..., ::-1], "bgr8")
        rgb_points_img.header = camera_info.header
        self.pub_rgb_dope_points.publish(rgb_points_img)
        self.pub_camera_info.publish(camera_info)
        self.pub_detections.publish(detection_array)
        self.publish_markers(detection_array)


    def publish_markers(self, detection_array):
        # # Object markers
        class_id_to_name = {class_id: name for name,
                            class_id in self.class_ids.items()}
        markers = MarkerArray()
        for i, det in enumerate(detection_array.detections):
            class_id = int(det.results[0].hypothesis.class_id)
            name = class_id_to_name.get(class_id)
            color = self.draw_colors[name]

            # cube marker
            marker = Marker()
            marker.header = detection_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.4
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox.size
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = detection_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 1.0
            marker.id = i
            marker.ns = "texts"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.text = '{} ({:.2f})'.format(name, det.results[0].hypothesis.score)
            markers.markers.append(marker)

            # mesh marker
            try:
                marker = Marker()
                marker.header = detection_array.header
                marker.action = Marker.ADD
                marker.pose = det.bbox.center
                marker.color.r = color[0] / 255.0
                marker.color.g = color[1] / 255.0
                marker.color.b = color[2] / 255.0
                marker.color.a = 0.7
                marker.ns = "meshes"
                marker.id = i
                marker.type = Marker.MESH_RESOURCE
                marker.scale.x = self.mesh_scales[name]
                marker.scale.y = self.mesh_scales[name]
                marker.scale.z = self.mesh_scales[name]
                marker.mesh_resource = self.meshes[name]
                markers.markers.append(marker)
            except KeyError:
                # user didn't specify self.meshes[name], so don't publish marker
                pass

        for i in range(len(detection_array.detections), self.prev_num_detections):
            for ns in ["bboxes", "texts", "meshes"]:
                marker = Marker()
                marker.action = Marker.DELETE
                marker.ns = ns
                marker.id = i
                markers.markers.append(marker)
        self.prev_num_detections = len(detection_array.detections)
        self.pub_markers.publish(markers)


def rotate_vector(vector, quaternion):

    # import tf_transformations
    q_conj = transformations.quaternion_conjugate(quaternion)
    vector = np.array(vector, dtype='float64')
    vector = np.append(vector, [0.0])
    vector = transformations.quaternion_multiply(q_conj, vector)
    vector = transformations.quaternion_multiply(vector, quaternion)
    
    return vector[:3]

def main():
    """Main routine to run DOPE"""
    
    # Initialize ROS node
    rclpy.init()
    
    dope_node = DopeNode()

    try:
        rclpy.spin(dope_node)
        pass

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")

    finally:
        dope_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
