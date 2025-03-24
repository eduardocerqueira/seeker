#date: 2025-03-24T17:06:22Z
#url: https://api.github.com/gists/5bbd25c681c230f3157bec7a76c97803
#owner: https://api.github.com/users/CarlosGTrejo

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from px4_msgs.msg import VehicleLocalPosition


qos_profile = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT  # Matches the publisher's reliability
)

def get_message_name_version(msg_class):
    if msg_class.MESSAGE_VERSION == 0:
        return ""
    return f"_v{msg_class.MESSAGE_VERSION}"


class AltitudeSubscriber(Node):
    def __init__(self):
        super().__init__('atitude_subscriber')

        sub_topic = f"/fmu/out/vehicle_local_position{get_message_name_version(VehicleLocalPosition)}"
        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            sub_topic,
            self.listener_callback,
            qos_profile
        )
        self.subscription

    def listener_callback(self, msg):
        altitude = -msg.z
        # In a NED coordinate system, z is down. To get a positive altitude (above ground), take the negative of z.
        print(f"Altitude: {altitude}", end='\r')


def main(args=None):
    rclpy.init(args=args)
    altitude_subscriber = AltitudeSubscriber()
    try:
        rclpy.spin(altitude_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        altitude_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()