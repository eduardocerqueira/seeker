#date: 2021-12-10T16:55:39Z
#url: https://api.github.com/gists/fcbafd5cf748c9b11e64a4dd37ec8e9a
#owner: https://api.github.com/users/papr

import logging
import time

import click
import msgpack
import zmq

logger = logging.getLogger(__name__)


@click.command()
@click.option("--ip", default="127.0.0.1", help="Pupil Capture or Service address")
@click.option("--port", default=50020, help="Pupil Remote port")
def main(ip, port):
    # Connect to Pupil Remote
    control_socket = pupil_remote_connection(ip, port)
    sub_socket = get_calibration_notification_socket(control_socket, ip)

    # Define local clock -
    # Required to calculate accurate timestamps for the reference data
    clock_function = time.monotonic
    # Get offset between local clock and Pupil clock
    clock_offset = _measure_clock_offset(control_socket, clock_function)
    logger.info(f"Measured clock offset: {clock_offset} seconds")

    # If the specific gazer class name is not found, Pupil Capture will fall back
    # to the default gazer class (3D calibration)
    gazer_class_name = "GazerDualDisplay2D"
    logger.info(
        "Enable ExternalCalibrationChoreography plugin with gazer class "
        + gazer_class_name
    )
    enable_external_choreography_plugin_selecting_gazer(
        control_socket, gazer_class_name
    )

    # Wait for user confirmation
    input("Press enter to start calibration")

    # before starting the calibration, make sure that no older calibration notifications
    # are queued in the socket. This is especially important if you want to repeat the
    # calibration with the same subscription socket.
    clear_socket_queue(sub_socket)

    # Start calibration
    control_socket.send_string("C")  # upper case C
    control_socket.recv_string()  # required as part of the zmq REQ-REP protocol
    wait_for_calibration_notification(sub_socket, "started")

    perform_choreography(control_socket, clock_function, clock_offset)

    # Stop calibration
    control_socket.send_string("c")  # lower case c
    control_socket.recv_string()  # required as part of the zmq REQ-REP protocol
    feedback = wait_for_calibration_notification(sub_socket, "successful", "failed")
    if feedback["subject"].endswith("failed"):
        logger.error(f"Calibration failed: {feedback['reason']}")
        raise SystemExit(1)

    # Calibration successful.
    # Create a new subscription socket, subscribe to `gaze`, and start processing
    # the incoming data in real time.


def pupil_remote_connection(ip, port):
    ctx = zmq.Context.instance()
    request_socket = ctx.socket(zmq.REQ)
    request_url = f"tcp://{ip}:{port}"
    logger.info(f"Connecting to {request_url}")
    request_socket.connect(request_url)
    return request_socket


def get_calibration_notification_socket(control_socket, ip):
    control_socket.send_string("SUB_PORT")
    sub_port = control_socket.recv_string()
    sub_url = f"tcp://{ip}:{sub_port}"
    ctx = zmq.Context.instance()
    sub_socket = ctx.socket(zmq.SUB)
    logger.info(f"Subscribing to {sub_url}")
    sub_socket.connect(sub_url)
    sub_socket.subscribe("notify.calibration")
    return sub_socket


def wait_for_calibration_notification(sub_socket, *topic_suffixes):
    while True:
        topic, notification = _recv_notification(sub_socket)
        if any(topic.endswith(suffix) for suffix in topic_suffixes):
            return notification
        else:
            logger.debug(f"Ignoring notification: {topic}")


def clear_socket_queue(socket):
    """Drop all messages in the socket's queue"""
    while socket.get(zmq.EVENTS) & zmq.POLLIN:
        try:
            socket.recv(zmq.NOBLOCK)
        except zmq.ZMQError:
            break


def enable_external_choreography_plugin_selecting_gazer(
    request_socket, gazer_class_name
):
    """Starts the calibration plugin"""
    notification = {
        "subject": "start_plugin",
        "name": "ExternalCalibrationChoreography",
        "args": {"selected_gazer_class_name": gazer_class_name},
    }
    _send_notification(request_socket, notification)


def perform_choreography(control_socket, clock_function, clock_offset):
    # example locations in normalized coordinates
    locations = (
        ([0.0, 1.0], "top left"),
        ([1.0, 1.0], "top right"),
        ([1.0, 0.0], "bottom right"),
        ([0.0, 0.0], "bottom left"),
        ([0.5, 0.5], "center"),
    )
    duration_per_location_seconds = 1.0
    for location_coord, location_human in locations:
        # To simulate a visual stimulus, the subject is instructed to look at a specific
        # location within their field of view for a specific duration. Meanwhile, the
        # code below generates reference locations at a specific sampling rate and sends
        # them to the Pupil Core software.
        # Another choreography client could display a visual stimulus, collect
        # timestamps after each frame, and send the reference data in bulk afterward
        _instruct_subject(location_human, duration_per_location_seconds)
        ref_data = []
        for _ in _timer(duration_per_location_seconds):
            # Add reference data for both eyes. Coordinates can differ between the two.
            ref_data.append(
                {
                    "norm_pos": location_coord,
                    "timestamp": clock_function() + clock_offset,
                    "eye_id": 0,  # right eye
                }
            )
            ref_data.append(
                {
                    "norm_pos": location_coord,
                    "timestamp": clock_function() + clock_offset,
                    "eye_id": 1,  # left eye
                }
            )
        _send_notification(
            control_socket,
            {"subject": "calibration.add_ref_data", "ref_data": ref_data},
        )


def _instruct_subject(target_location_humand_description, duration_seconds):
    input(
        f"Look to the {target_location_humand_description}, hit enter, and keep looking"
        f" at the target location for {duration_seconds} seconds"
    )


def _timer(duration_seconds=1.0, sampling_rate_hz=30):
    """Returns control at a fixed rate for `duration_seconds`"""
    num_samples = int(duration_seconds * sampling_rate_hz)
    duration_between_samples = duration_seconds / sampling_rate_hz
    for i in range(num_samples):
        yield
        time.sleep(duration_between_samples)


def _send_notification(request_socket, notification):
    """Sends ``notification`` to Pupil Remote"""
    topic = "notify." + notification["subject"]
    payload = msgpack.dumps(notification, use_bin_type=True)
    request_socket.send_string(topic, flags=zmq.SNDMORE)
    request_socket.send(payload)
    return request_socket.recv_string()


def _recv_notification(sub_socket):
    """Receives a notification from Pupil Remote"""
    topic = sub_socket.recv_string()
    payload = sub_socket.recv()
    notification = msgpack.unpackb(payload)
    return topic, notification


def _measure_clock_offset(request_socket, clock_function):
    """Calculates the offset between the Pupil Core software clock and a local clock.
    Requesting the remote pupil time takes time. This delay needs to be considered
    when calculating the clock offset. We measure the local time before (A) and
    after (B) the request and assume that the remote pupil time was measured at (A+B)/2,
    i.e. the midpoint between A and B.
    As a result, we have two measurements from two different clocks that were taken
    assumingly at the same point in time. The difference between them ("clock offset")
    allows us, given a new local clock measurement, to infer the corresponding time on
    the remote clock.

    See this helper for reference:
    https://github.com/pupil-labs/pupil-helpers/blob/master/python/simple_realtime_time_sync.py
    """
    local_time_before = clock_function()
    request_socket.send_string("t")
    pupil_time = float(request_socket.recv_string())
    local_time_after = clock_function()

    local_time = (local_time_before + local_time_after) / 2.0
    clock_offset = pupil_time - local_time
    return clock_offset


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
