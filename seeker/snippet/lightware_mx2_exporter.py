#date: 2025-04-03T16:56:31Z
#url: https://api.github.com/gists/0c2ce612273db11ad9d8edcd0643ab44
#owner: https://api.github.com/users/llamafilm

import logging
import os
import urllib
import socket
import sys
import wsgiref

import prometheus_client
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily


class LightwareMX2Exporter(prometheus_client.registry.Collector):
    def __init__(self, target: str):
        self.target = target
        self.port = 6107

    def to_binary(self, str: str) -> int:
        """Convert string representation of bool or 0/1 to integer"""
        if str.lower() in ["true", "1"]:
            return 1
        else:
            return 0

    def get(self, cmd: str) -> str:
        """Send command to device and return parsed response"""
        self.sock.sendall(f"GET {cmd}\r\n".encode())

        response = ""
        for _ in range(100):
            data = self.sock.recv(1024)
            response += data.decode()
            if data.decode().endswith("\r\n") or data.decode() == "":
                break

        return response.strip().split(cmd)[1][1:]

    def get_nodes(self, cmd: str) -> list:
        """Send command to device and return list of nodes"""
        self.sock.sendall(f"GET {cmd}\r\n".encode())

        response = ""
        for _ in range(100):
            data = self.sock.recv(1024)
            response += data.decode()
            if data.decode().endswith("\r\n") or data.decode() == "":
                break

        results = []
        for line in response.strip().split("\r\n"):
            results.append(line.split(cmd)[1][1:])

        return results

    def get_properties(self, cmd: str) -> list:
        """Send command to device and return list of properties"""
        signature = os.urandom(2).hex()
        self.sock.sendall(f"{signature}#GET {cmd}.*\r\n".encode())

        response = ""
        for _ in range(100):
            data = self.sock.recv(1024)
            response += data.decode()
            if data.decode().endswith("}\r\n") or data.decode() == "":
                break

        results = {}
        for line in response.strip()[7:-3].split("\r\n"):
            try:
                key = line.split(cmd)[1][1:].split("=")[0]
                value = line.split(cmd)[1][1:].split("=")[1] or None
                results[key] = value
            except IndexError:
                continue
        return results

    def describe(self):
        return[]

    def collect(self):
        print("collecting")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect((self.target, self.port))

        # read CPU temperature
        fan_response = self.get_properties("/SYS/HSMB/FANCONTROL")
        matrix_temperature = GaugeMetricFamily("mx2_matrix_temperature", "Temperature of the hottest part of the system in celsius. /SYS/HSMB/FANCONTROL")
        matrix_temperature.add_metric([], float(fan_response["MaximalCurrentTemperature"]))
        yield matrix_temperature

        # read uptime
        datetime_response = self.get_properties("/MANAGEMENT/DATETIME")
        days, time_str = datetime_response["Uptime"].split(" days ")
        hours, minutes, seconds = map(int, time_str.split(":"))
        uptime = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
        matrix_uptime_seconds = GaugeMetricFamily("mx2_matrix_uptime_seconds", "Uptime of the core software, in seconds. /MANAGEMENT/DATETIME")
        matrix_uptime_seconds.add_metric([], uptime)
        yield matrix_uptime_seconds

        # read average of 3 fan speeds
        avg_fan_rpm = (
            (float(fan_response["Fan1Pwm"]) + float(fan_response["Fan2Pwm"]) + float(fan_response["Fan3Pwm"])) / 3 / 255
        )
        matrix_fan_percent = GaugeMetricFamily("mx2_matrix_fan_percent", "Average fan speed as a percentage of maximum")
        matrix_fan_percent.add_metric([], avg_fan_rpm)
        yield matrix_fan_percent

        # read firmware version
        uid_response = self.get_properties("/MANAGEMENT/UID")
        matrix_info = InfoMetricFamily("mx2_matrix_info","Info metric with a constant '1' value labeled by serial and firmware_version")
        matrix_info.add_metric(labels=[], value={
            "serial": uid_response["ProductSerialNumber"],
            "firmware_version": uid_response["FirmwareVersion"],
        })
        yield matrix_info

        # read names of each port
        response = self.get_properties("/MEDIA/NAMES/VIDEO")
        port_names = {}
        for port in response:
            port_names[port] = response[port].split(";")[1]

        # metric for which input is routed to each destination
        # value of 0 means no source is routed
        response = self.get("/MEDIA/XP/VIDEO.DestinationConnectionStatus")
        xpt_connection = GaugeMetricFamily(
            "mx2_xpt_connection",
            "Source number currently routed to the destination. /MEDIA/XP/VIDEO.DestinationConnectionStatus",
            labels=["port", "port_name"],
        )
        for i, value in enumerate(response.split(";")[:-1]):
            port = f"O{i + 1}"
            if value == "0":
                value = "00"
            xpt_connection.add_metric([port, port_names[port]], int(value[1:]))
        yield xpt_connection

        # metric for destination port locked/muted state
        response = self.get("/MEDIA/XP/VIDEO.DestinationPortStatus")
        dest_locked = GaugeMetricFamily(
            "mx2_dest_locked",
            "Boolean value 0 or 1. /MEDIA/XP/VIDEO.DestinationPortStatus",
            labels=["port", "port_name"],
        )
        dest_muted = GaugeMetricFamily(
            "mx2_dest_muted",
            "Boolean value 0 or 1. /MEDIA/XP/VIDEO.DestinationPortStatus",
            labels=["port", "port_name"],
        )
        for i, code in enumerate(response.split(";")[:-1]):
            port = f"O{i + 1}"
            dest_locked.add_metric([port, port_names[port]], 0 if code[0] in ["T", "M"] else 1)
            dest_muted.add_metric([port, port_names[port]], 0 if code[0] in ["T", "L"] else 1)
        yield dest_locked
        yield dest_muted

        # metric for source port locked/muted state
        response = self.get("/MEDIA/XP/VIDEO.SourcePortStatus")
        src_locked = GaugeMetricFamily(
            "mx2_src_locked",
            "Boolean value 0 or 1. /MEDIA/XP/VIDEO.SourcePortStatus",
            labels=["port", "port_name"],
        )
        src_muted = GaugeMetricFamily(
            "mx2_src_muted",
            "Boolean value 0 or 1. /MEDIA/XP/VIDEO.SourcePortStatus",
            labels=["port", "port_name"],
        )
        for i, code in enumerate(response.split(";")[:-1]):
            port = f"I{i + 1}"
            src_locked.add_metric([port, port_names[port]], 0 if code[0] in ["T", "M"] else 1)
            src_muted.add_metric([port, port_names[port]], 0 if code[0] in ["T", "L"] else 1)

        yield src_locked
        yield src_muted

        # metrics for properties of each port
        port_connected = GaugeMetricFamily(
            "mx2_port_connected",
            "Boolean 0 or 1 indicates whether cable +5V is present or not",
            labels=["port", "port_name"],
        )
        port_active_hdcp_version = GaugeMetricFamily(
            "mx2_port_active_hdcp_version",
            "0=off, 1=HDCP1.4, 2=HDCP2.2",
            labels=["port", "port_name"],
        )
        port_color_depth = GaugeMetricFamily(
            "mx2_port_color_depth",
            "Color depth per channel (usually between 8 and 16)",
            labels=["port", "port_name"],
        )
        port_embedded_audio_present = GaugeMetricFamily(
            "mx2_port_embedded_audio_present",
            "Boolean 0 or 1 indicates the presence of embedded audio",
            labels=["port", "port_name"],
        )
        port_hdcp2_stream_type = GaugeMetricFamily(
            "mx2_port_hdcp2_stream_type",
            "HDCP Stream Type for streams with HDCP2.2. In case of HDCP1.4 the value is ignored.",
            labels=["port", "port_name"],
        )
        port_max_supported_hdcp_version = GaugeMetricFamily(
            "mx2_port_max_supported_hdcp_version",
            "0=None, 1=HDCP1.4, 2=HDCP2.2",
            labels=["port", "port_name"],
        )
        port_pixel_clock = GaugeMetricFamily(
            "mx2_port_pixel_clock",
            "Frequency of the pixel clock in MHz",
            labels=["port", "port_name"],
            unit="mhz",
        )
        port_scrambling = GaugeMetricFamily(
            "mx2_port_scrambling",
            "Boolean 0 or 1 indicates HDMI scrambling status",
            labels=["port", "port_name"],
        )
        port_tmds_clock_rate = GaugeMetricFamily(
            "mx2_port_tmds_clock_rate",
            "TMDS clock rate, 0=1/10, 1=1/40",
            labels=["port", "port_name"],
        )
        port_bch_error_count = GaugeMetricFamily(
            "mx2_port_bch_error_count",
            "BCH ECC error counter (max 32)",
            labels=["port", "port_name"],
        )
        port_signal_present = GaugeMetricFamily(
            "mx2_port_signal_present",
            "Boolean 0 or 1 indicates valid signal present on the port",
            labels=["port", "port_name"],
        )
        port_tmds_error_count = GaugeMetricFamily(
            "mx2_port_tmds_error_count",
            "TMDS error counters for each of 3 channels",
            labels=["port", "port_name", "tmds_ch"],
        )
        port_rx_tmds_error_count = GaugeMetricFamily(
            "mx2_port_rx_tmds_error_count",
            "Rx TMDS error counters for each of 3 channels",
            labels=["port", "port_name", "tmds_ch"],
        )
        port_active_resolution = InfoMetricFamily(
            "mx2_port_active_resolution", "The resolution of the signal based on the AVI InfoFrame"
        )
        port_total_resolution = InfoMetricFamily(
            "mx2_port_total_resolution",
            "The resolution of the video signal with blanking based on the AVI InfoFrame",
        )
        port_color_space = InfoMetricFamily(
            "mx2_port_color_space",
            "Color space of the signal: RGB, YUV_444, YUV_422, YUV_420, or UNKNOWN",
        )
        port_color_range = InfoMetricFamily(
            "mx2_port_color_range", "Color range of the signal: FULL, LIMITED, or UNKNOWN"
        )
        port_signal_type = InfoMetricFamily(
            "mx2_port_signal_type", "Indicates signal type of the video: DVI, HDMI, DP, SDI, VGA"
        )
        port_avi_if = InfoMetricFamily("mx2_port_avi_if", "HDMI AVI InfoFrame [*{2_hex_octet}]")
        port_vs_if = InfoMetricFamily("mx2_port_vs_if", "HDMI Vendor Specific InfoFrame [*{2_hex_octet}]")

        ports = self.get_nodes("/MEDIA/PORTS/VIDEO")
        port_labels = [port, port_names[port]]
        for port in ports:
            status = self.get_properties(f"/MEDIA/PORTS/VIDEO/{port}/STATUS")
            port_connected.add_metric(port_labels, 0 if status["Connected"].lower() == "false" else 1)
            port_active_hdcp_version.add_metric(port_labels, int(status["ActiveHdcpVersion"]))
            port_color_depth.add_metric(port_labels, int(status["ColorDepth"]))
            port_embedded_audio_present.add_metric(
                port_labels, self.to_binary(status["EmbeddedAudioPresent"])
            )
            port_hdcp2_stream_type.add_metric(port_labels, int(status["Hdcp2StreamType"]))
            port_max_supported_hdcp_version.add_metric(port_labels, int(status["MaxSupportedHdcpVersion"]))
            port_pixel_clock.add_metric(port_labels, float(status["PixelClock"]))
            port_scrambling.add_metric(port_labels, self.to_binary(status["Scrambling"]))
            port_tmds_clock_rate.add_metric(port_labels, self.to_binary(status["TmdsClockRate"]))
            port_bch_error_count.add_metric(port_labels, int(status["BchErrorCounter"]))
            port_signal_present.add_metric(port_labels, self.to_binary(status["SignalPresent"]))
            for ch in range(3):
                port_tmds_error_count.add_metric(
                    [port, port_names[port], str(ch)],
                    self.to_binary(status["TmdsErrorCounters"].split(";")[ch]),
                )
                port_rx_tmds_error_count.add_metric(
                    [port, port_names[port], str(ch)],
                    self.to_binary(status["RxTmdsErrorCounters"].split(";")[ch]),
                )

            port_active_resolution.add_metric(
                labels=[],
                value={
                    "port": port,
                    "port_name": port_names[port],
                    "port_active_resolution": str(status["ActiveResolution"]),
                },
            )
            port_total_resolution.add_metric(
                labels=[],
                value={
                    "port": port,
                    "port_name": port_names[port],
                    "port_total_resolution": str(status["TotalResolution"]),
                },
            )
            port_color_space.add_metric(
                labels=[],
                value={
                    "port": port,
                    "port_name": port_names[port],
                    "port_color_space": str(status["ColorSpace"]),
                },
            )
            port_color_range.add_metric(
                labels=[],
                value={
                    "port": port,
                    "port_name": port_names[port],
                    "port_color_range": str(status["ColorRange"]),
                },
            )
            port_signal_type.add_metric(
                labels=[],
                value={
                    "port": port,
                    "port_name": port_names[port],
                    "port_signal_type": str(status["SignalType"]),
                },
            )
            port_avi_if.add_metric(
                labels=[],
                value={
                    "port": port,
                    "port_name": port_names[port],
                    "port_avi_if": str(status["AviIf"]),
                },
            )
            port_vs_if.add_metric(
                labels=[],
                value={
                    "port": port,
                    "port_name": port_names[port],
                    "port_vs_if": str(status["VsIf"]),
                },
            )

        yield port_connected
        yield port_active_hdcp_version
        yield port_color_depth
        yield port_embedded_audio_present
        yield port_hdcp2_stream_type
        yield port_max_supported_hdcp_version
        yield port_pixel_clock
        yield port_scrambling
        yield port_tmds_clock_rate
        yield port_bch_error_count
        yield port_signal_present
        yield port_tmds_error_count
        yield port_rx_tmds_error_count
        yield port_active_resolution
        yield port_total_resolution
        yield port_color_space
        yield port_color_range
        yield port_signal_type
        yield port_avi_if
        yield port_vs_if

        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()


def wsgi_app(environ, start_response):
    path = wsgiref.util.shift_path_info(environ)
    if path == "probe":
        try:
            qs = urllib.parse.parse_qs(environ["QUERY_STRING"])
            target = qs["target"][0]
        except KeyError:
            start_response("400 Bad Request", [("Content-Type", "text/plain")])
            return [b"Target parameter is missing\r\n"]

        registry = prometheus_client.registry.CollectorRegistry(auto_describe=False)
        registry.register(LightwareMX2Exporter(target))
        prometheus_app = prometheus_client.make_wsgi_app(registry)
        return prometheus_app(environ, start_response)

    else:
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found. Use /probe endpoint.\r\n"]


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    port = int(sys.argv[1])
    httpd = wsgiref.simple_server.make_server("", port, wsgi_app)
    print(f"Serving on port {port}...")
    httpd.serve_forever()
