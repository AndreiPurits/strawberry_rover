import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraPreprocessNode(Node):
    """Subscribes to RGB images and publishes simple preprocessed outputs."""

    def __init__(self) -> None:
        super().__init__("camera_preprocess_node")

        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("gray_topic", "/camera/preprocessed/gray")
        self.declare_parameter("edges_topic", "/camera/preprocessed/edges")
        self.declare_parameter("blur_kernel_size", 5)
        self.declare_parameter("canny_threshold_low", 60)
        self.declare_parameter("canny_threshold_high", 140)

        input_topic = str(self.get_parameter("input_topic").value)
        gray_topic = str(self.get_parameter("gray_topic").value)
        edges_topic = str(self.get_parameter("edges_topic").value)

        self._blur_kernel_size = int(self.get_parameter("blur_kernel_size").value)
        self._canny_low = int(self.get_parameter("canny_threshold_low").value)
        self._canny_high = int(self.get_parameter("canny_threshold_high").value)

        if self._blur_kernel_size < 1:
            self._blur_kernel_size = 1
        if self._blur_kernel_size % 2 == 0:
            self._blur_kernel_size += 1

        self._sub = self.create_subscription(
            Image, input_topic, self._on_image, qos_profile_sensor_data
        )
        self._gray_pub = self.create_publisher(Image, gray_topic, qos_profile_sensor_data)
        self._edges_pub = self.create_publisher(Image, edges_topic, qos_profile_sensor_data)

        self.get_logger().info(
            "camera_preprocess_node started. "
            f"input={input_topic}, gray={gray_topic}, edges={edges_topic}"
        )

    def _on_image(self, msg: Image) -> None:
        frame = self._from_image_msg(msg)
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self._blur_kernel_size, self._blur_kernel_size), 0)
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)

        self._gray_pub.publish(self._to_mono8_msg(gray, msg))
        self._edges_pub.publish(self._to_mono8_msg(edges, msg))

    def _from_image_msg(self, msg: Image):
        if msg.encoding != "bgr8":
            self.get_logger().warn(
                f"Unsupported encoding '{msg.encoding}', expected 'bgr8'."
            )
            return None

        if msg.height <= 0 or msg.width <= 0:
            self.get_logger().warn("Received invalid image dimensions.")
            return None

        np_frame = np.frombuffer(msg.data, dtype=np.uint8)
        expected_size = int(msg.height) * int(msg.width) * 3
        if np_frame.size != expected_size:
            self.get_logger().warn("Received image with unexpected data size.")
            return None

        return np_frame.reshape((int(msg.height), int(msg.width), 3))

    def _to_mono8_msg(self, frame: np.ndarray, src: Image) -> Image:
        out = Image()
        out.header = src.header
        out.height = int(frame.shape[0])
        out.width = int(frame.shape[1])
        out.encoding = "mono8"
        out.is_bigendian = 0
        out.step = int(frame.shape[1])
        out.data = frame.tobytes()
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraPreprocessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
