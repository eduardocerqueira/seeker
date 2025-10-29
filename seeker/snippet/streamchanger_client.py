#date: 2025-10-29T16:59:25Z
#url: https://api.github.com/gists/4c7fc315d4e18244761efe02810810a3
#owner: https://api.github.com/users/esmaili417

#!/usr/bin/env python3
"""
StreamChanger Pi Client v2.1.0
Connects Raspberry Pi to StreamChanger web app for RTSP-to-RTMPS conversion
With automatic reconnection and remote management capabilities
"""

import os
import sys
import time
import json
import requests
import subprocess
import base64
import websocket
import threading
import shutil
import re
from datetime import datetime
from pathlib import Path
import socket
import concurrent.futures

# Client version
CLIENT_VERSION = "2.1.0"

# Load configuration
config_file = Path.home() / '.streamchanger_config'
config = {}

if config_file.exists():
    with open(config_file) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                config[key] = value.strip('"')

API_URL = config.get('STREAMCHANGER_URL', 'http://localhost:5000')
API_KEY = config.get('API_KEY', '')
POLL_INTERVAL = int(config.get('POLL_INTERVAL', 30))

if not API_KEY:
    print("ERROR: API_KEY not found in ~/.streamchanger_config")
    sys.exit(1)

# Headers for API authentication
HEADERS = {'X-API-Key': API_KEY}

# Active FFmpeg processes
active_streams = {}

# Device info (cached after authentication)
device_info = None

# WebSocket connection
ws = None
ws_connected = False
should_reconnect = True  # Flag to control reconnection behavior


def authenticate_device():
    """Verify API key and get device info"""
    global device_info
    try:
        response = requests.get(f"{API_URL}/api/pi/auth", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            device_info = response.json()
            print(f"[SUCCESS] Authenticated as: {device_info['name']}")
            print(f"  Client Version: {CLIENT_VERSION}")
            if device_info.get('rtspUrl'):
                print(f"  RTSP Source: {device_info['rtspUrl']}")
            else:
                print("  [WARNING] No RTSP URL configured for this device")
            return device_info
        else:
            print(f"[ERROR] Authentication failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Connection error: {e}")
        return None


def get_stream_configs():
    """Fetch stream configurations from the server"""
    try:
        response = requests.get(f"{API_URL}/api/pi/streams", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"Error fetching streams: {e}")
        return []


def get_active_schedules():
    """Fetch currently active schedules"""
    try:
        response = requests.get(f"{API_URL}/api/pi/schedules", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"Error fetching schedules: {e}")
        return []


def update_device_status(status):
    """Update device status on server"""
    try:
        requests.post(
            f"{API_URL}/api/pi/status",
            headers=HEADERS,
            json={"status": status},
            timeout=5
        )
    except Exception as e:
        print(f"Error updating status: {e}")


def update_stream_status(stream_id, status):
    """Update status of a specific stream"""
    try:
        requests.post(
            f"{API_URL}/api/pi/streams/{stream_id}/status",
            headers=HEADERS,
            json={"status": status},
            timeout=5
        )
    except Exception as e:
        print(f"Error updating stream status: {e}")


def send_metrics(stream_id, device_id, metrics):
    """Send stream performance metrics to server"""
    try:
        data = {
            "streamId": stream_id,
            "deviceId": device_id,
            "bitrate": metrics.get("bitrate", "0"),
            "fps": metrics.get("fps", "0"),
            "errorCount": metrics.get("errorCount", "0"),
            "packetLoss": metrics.get("packetLoss", "0"),
            "droppedFrames": metrics.get("droppedFrames", "0"),
            "cpuUsage": metrics.get("cpuUsage", "0"),
            "memoryUsage": metrics.get("memoryUsage", "0")
        }
        requests.post(f"{API_URL}/api/pi/metrics", headers=HEADERS, json=data, timeout=5)
    except Exception as e:
        print(f"Error sending metrics: {e}")


def start_stream(stream_config):
    """Start RTSP to RTMPS conversion using FFmpeg"""
    stream_id = stream_config['id']

    # Get RTSP URL from device (not from stream)
    if not device_info or not device_info.get('rtspUrl'):
        print(f"[ERROR] Cannot start stream: No RTSP URL configured for device")
        update_stream_status(stream_id, 'error')
        return

    rtsp_url = device_info['rtspUrl']
    rtmps_url = stream_config['rtmpsUrl']
    rtmps_key = stream_config.get('rtmpsKey', '')

    # Build full RTMPS URL with key if provided
    if rtmps_key:
        full_rtmps_url = f"{rtmps_url}/{rtmps_key}"
    else:
        full_rtmps_url = rtmps_url

    if stream_id in active_streams:
        print(f"Stream {stream_id} already running")
        return

    print(f"[STREAM START] Starting stream: {stream_config['name']}")
    print(f"  RTSP: {rtsp_url}")
    print(f"  RTMPS: {full_rtmps_url}")

    # FFmpeg command for RTSP to RTMPS conversion
    cmd = [
        'ffmpeg',
        '-i', rtsp_url,           # Input RTSP stream
        '-c:v', 'copy',            # Copy video codec (no re-encoding)
        '-c:a', 'aac',             # Audio codec
        '-f', 'flv',               # Output format for RTMPS
        full_rtmps_url             # Output RTMPS URL (with key if provided)
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        active_streams[stream_id] = {
            'process': process,
            'config': stream_config,
            'start_time': datetime.now()
        }
        update_stream_status(stream_id, 'active')
        print(f"[SUCCESS] Stream started: {stream_config['name']}")
    except Exception as e:
        print(f"[ERROR] Failed to start stream: {e}")
        update_stream_status(stream_id, 'error')


def stop_stream(stream_id):
    """Stop an active stream"""
    if stream_id not in active_streams:
        print(f"Stream {stream_id} not running")
        return

    stream_name = active_streams[stream_id]['config'].get('name', stream_id[:8])
    print(f"[STREAM STOP] Stopping stream: {stream_name}")
    stream_info = active_streams[stream_id]
    stream_info['process'].terminate()

    try:
        stream_info['process'].wait(timeout=5)
    except subprocess.TimeoutExpired:
        stream_info['process'].kill()

    del active_streams[stream_id]
    update_stream_status(stream_id, 'inactive')
    print(f"[SUCCESS] Stream stopped: {stream_name}")


def sync_streams():
    """Sync running streams with server's active stream list"""
    # Get what the server says should be running
    server_streams = get_stream_configs()
    server_stream_ids = {s['id'] for s in server_streams}

    # Get what's currently running locally
    local_stream_ids = set(active_streams.keys())

    # Stop streams that are no longer active on server
    streams_to_stop = local_stream_ids - server_stream_ids
    for stream_id in streams_to_stop:
        stop_stream(stream_id)

    # Start streams that should be running but aren't
    for stream in server_streams:
        if stream['id'] not in active_streams and stream.get('isEnabled', True):
            start_stream(stream)


def scan_network_for_cameras(timeout=10):
    """
    Scan local network for cameras using port scanning
    More reliable than ONVIF multicast discovery
    """
    discovered_cameras = []
    
    try:
        print("[CAMERA SCAN] Starting network scan...")
        
        # Get local network
        result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
        network_base = None
        
        for line in result.stdout.split('\n'):
            if 'src' in line:
                match = re.search(r'src\s+(\d+\.\d+\.\d+\.\d+)', line)
                if match:
                    local_ip = match.group(1)
                    network_base = '.'.join(local_ip.split('.')[:-1])
                    break
        
        if not network_base:
            network_base = '192.168.1'
        
        network = f"{network_base}.0/24"
        print(f"[CAMERA SCAN] Scanning network: {network}")
        
        # Quick nmap scan for live hosts
        cmd = ['nmap', '-sn', '-T4', network]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        # Extract IPs
        ips = re.findall(r'\d+\.\d+\.\d+\.\d+', result.stdout)
        print(f"[CAMERA SCAN] Found {len(ips)} active devices, testing for cameras...")
        
        # Test each IP for camera ports in parallel
        def test_ip(ip):
            open_ports = []
            for port in [80, 554, 8080]:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.5)
                    result = sock.connect_ex((ip, port))
                    sock.close()
                    if result == 0:
                        open_ports.append(port)
                except:
                    pass
            
            # If has port 80 or 554, likely a camera
            if 80 in open_ports or 554 in open_ports:
                return {
                    'ip': ip,
                    'open_ports': open_ports,
                    'model': 'IP Camera',
                    'manufacturer': 'Unknown'
                }
            return None
        
        # Scan IPs in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(test_ip, ip): ip for ip in ips}
            
            for future in concurrent.futures.as_completed(futures):
                camera_info = future.result()
                if camera_info:
                    discovered_cameras.append(camera_info)
                    print(f"[CAMERA SCAN] Found: {camera_info['ip']} (ports: {camera_info['open_ports']})")
        
        print(f"[CAMERA SCAN] Complete - found {len(discovered_cameras)} camera(s)")
        
    except Exception as e:
        print(f"[CAMERA SCAN] Error: {e}")
    
    return discovered_cameras


def get_system_info():
    """Get device system information"""
    try:
        # Get CPU temperature (Raspberry Pi)
        temp = 'N/A'
        try:
            temp_cmd = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
            if temp_cmd.returncode == 0:
                temp = temp_cmd.stdout.strip()
        except:
            pass
        
        # Get disk usage
        disk_usage = 'N/A'
        try:
            disk_cmd = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
            disk_lines = disk_cmd.stdout.split('\n')
            if len(disk_lines) > 1:
                disk_usage = disk_lines[1].split()[4]
        except:
            pass
        
        # Get memory usage
        mem_usage = 'N/A'
        try:
            mem_cmd = subprocess.run(['free', '-h'], capture_output=True, text=True)
            mem_lines = mem_cmd.stdout.split('\n')
            if len(mem_lines) > 1:
                mem_usage = mem_lines[1].split()[2]
        except:
            pass
        
        # Get uptime
        uptime = 'N/A'
        try:
            uptime = subprocess.run(['uptime', '-p'], capture_output=True, text=True).stdout.strip()
        except:
            pass
        
        return {
            'version': CLIENT_VERSION,
            'temperature': temp,
            'diskUsage': disk_usage,
            'memoryUsage': mem_usage,
            'uptime': uptime,
            'activeStreams': len(active_streams)
        }
    except Exception as e:
        return {'error': str(e)}


def handle_snapshot_request(ws, request_id):
    """Capture a frame from RTSP stream and send as base64 JPEG"""
    try:
        # Get RTSP URL from device info
        if not device_info or not device_info.get('rtspUrl'):
            ws.send(json.dumps({
                'type': 'snapshot_response',
                'requestId': request_id,
                'success': False,
                'error': 'No RTSP URL configured'
            }))
            return

        rtsp_url = device_info['rtspUrl']
        print(f"[SNAPSHOT] Capturing snapshot from {rtsp_url}")

        # Use FFmpeg to capture single frame from RTSP stream
        ffmpeg_cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-frames:v', '1',           # Capture only 1 frame
            '-q:v', '2',                # JPEG quality (1-31, lower is better)
            '-f', 'image2pipe',         # Output to pipe
            '-vcodec', 'mjpeg',         # JPEG codec
            'pipe:1'                    # Output to stdout
        ]

        # Execute FFmpeg and capture output
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            timeout=5,  # 5 second timeout for capture
            check=False
        )

        if result.returncode == 0 and result.stdout:
            # Encode JPEG to base64
            image_base64 = base64.b64encode(result.stdout).decode('utf-8')

            # Send success response
            ws.send(json.dumps({
                'type': 'snapshot_response',
                'requestId': request_id,
                'success': True,
                'imageData': image_base64
            }))
            print(f"[SUCCESS] Snapshot captured and sent (requestId: {request_id})")
        else:
            # FFmpeg failed
            error_msg = result.stderr.decode('utf-8') if result.stderr else 'FFmpeg capture failed'
            ws.send(json.dumps({
                'type': 'snapshot_response',
                'requestId': request_id,
                'success': False,
                'error': f'Failed to capture frame: {error_msg[:200]}'
            }))
            print(f"[ERROR] Snapshot capture failed: {error_msg[:100]}")

    except subprocess.TimeoutExpired:
        ws.send(json.dumps({
            'type': 'snapshot_response',
            'requestId': request_id,
            'success': False,
            'error': 'Snapshot capture timed out after 5 seconds'
        }))
        print(f"[ERROR] Snapshot capture timed out (requestId: {request_id})")

    except Exception as e:
        ws.send(json.dumps({
            'type': 'snapshot_response',
            'requestId': request_id,
            'success': False,
            'error': f'Snapshot error: {str(e)}'
        }))
        print(f"[ERROR] Snapshot error: {e}")


def handle_firmware_update(ws, request_id, update_data):
    """Download and install firmware update"""
    try:
        print("[FIRMWARE] Update request received")
        
        # Get update URL from server
        update_url = update_data.get('updateUrl')
        version = update_data.get('version')
        
        if not update_url:
            ws.send(json.dumps({
                'type': 'firmware_update_response',
                'requestId': request_id,
                'success': False,
                'error': 'No update URL provided'
            }))
            return
        
        print(f"[FIRMWARE] Downloading version {version}...")
        
        # Download new client code
        response = requests.get(update_url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code}")
        
        # Backup current version
        backup_path = Path.home() / 'streamchanger_client.py.backup'
        current_path = Path.home() / 'streamchanger_client.py'
        
        if current_path.exists():
            shutil.copy(current_path, backup_path)
            print("[FIRMWARE] Current version backed up")
        
        # Write new version
        with open(current_path, 'w') as f:
            f.write(response.text)
        
        print("[FIRMWARE] New version installed")
        
        # Send success response
        ws.send(json.dumps({
            'type': 'firmware_update_response',
            'requestId': request_id,
            'success': True,
            'version': version
        }))
        
        # Restart service after 2 seconds
        print("[FIRMWARE] Restarting in 2 seconds...")
        time.sleep(2)
        subprocess.run(['sudo', 'systemctl', 'restart', 'streambridge'])
        
    except Exception as e:
        print(f"[FIRMWARE] Update failed: {e}")
        ws.send(json.dumps({
            'type': 'firmware_update_response',
            'requestId': request_id,
            'success': False,
            'error': str(e)
        }))


def handle_config_update(ws, request_id, config_data):
    """Update device configuration"""
    try:
        print("[CONFIG] Update request received")
        
        # Read current config
        current_config = {}
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        current_config[key] = value.strip('"')
        
        # Update with new values
        for key, value in config_data.items():
            current_config[key] = value
            print(f"[CONFIG] Set {key} = {value}")
        
        # Write updated config
        with open(config_file, 'w') as f:
            f.write("# StreamChanger Device Configuration\n")
            for key, value in current_config.items():
                f.write(f'{key}="{value}"\n')
        
        ws.send(json.dumps({
            'type': 'config_update_response',
            'requestId': request_id,
            'success': True
        }))
        
        print("[CONFIG] Configuration updated - restart required for changes to take effect")
        
    except Exception as e:
        print(f"[CONFIG] Update failed: {e}")
        ws.send(json.dumps({
            'type': 'config_update_response',
            'requestId': request_id,
            'success': False,
            'error': str(e)
        }))


def handle_restart_request(ws, request_id):
    """Restart the StreamBridge service"""
    try:
        print("[RESTART] Restart request received")
        
        ws.send(json.dumps({
            'type': 'restart_response',
            'requestId': request_id,
            'success': True,
            'message': 'Restarting in 2 seconds...'
        }))
        
        # Give time for message to send
        time.sleep(2)
        
        # Restart systemd service
        subprocess.run(['sudo', 'systemctl', 'restart', 'streambridge'])
        
    except Exception as e:
        print(f"[RESTART] Failed: {e}")
        ws.send(json.dumps({
            'type': 'restart_response',
            'requestId': request_id,
            'success': False,
            'error': str(e)
        }))


def on_ws_message(ws, message):
    """Handle WebSocket messages from server"""
    try:
        data = json.loads(message)
        msg_type = data.get('type')

        if msg_type == 'connected':
            print(f"[WEBSOCKET] Connected: {data.get('message')}")

        elif msg_type == 'heartbeat_ack':
            pass  # Heartbeat acknowledged

        elif msg_type == 'stream_start':
            # Immediate start command via WebSocket
            stream_id = data.get('streamId')
            print(f"[WEBSOCKET] START command for {stream_id[:8]}")
            sync_streams()  # Sync immediately

        elif msg_type == 'stream_stop':
            # Immediate stop command via WebSocket
            stream_id = data.get('streamId')
            print(f"[WEBSOCKET] STOP command for {stream_id[:8]}")
            if stream_id in active_streams:
                stop_stream(stream_id)

        elif msg_type == 'schedule_updated':
            print("[SCHEDULE] Schedule updated - syncing streams")
            sync_streams()

        elif msg_type == 'stream_updated':
            print(f"[STREAM] Stream {data.get('streamId')[:8]} updated - syncing")
            sync_streams()

        elif msg_type == 'snapshot_request':
            # Handle snapshot request from web app
            request_id = data.get('requestId')
            print(f"[SNAPSHOT] Request received (requestId: {request_id})")
            handle_snapshot_request(ws, request_id)

        elif msg_type == 'camera_discovery_request':
            request_id = data.get('requestId')
            print(f"[CAMERA DISCOVERY] Request received (ID: {request_id})")
            
            # Run network scan in background
            def run_discovery():
                try:
                    cameras = scan_network_for_cameras(timeout=10)
                    
                    # Format response - only include devices with RTSP port (554)
                    formatted_cameras = []
                    for cam in cameras:
                        if 554 in cam['open_ports']:
                            formatted_cameras.append({
                                'ip': cam['ip'],
                                'model': f"Camera at {cam['ip']}",
                                'manufacturer': 'Network Camera'
                            })
                    
                    # Send response
                    ws.send(json.dumps({
                        'type': 'camera_discovery_response',
                        'requestId': request_id,
                        'success': True,
                        'cameras': formatted_cameras
                    }))
                    print(f"[CAMERA DISCOVERY] Sent {len(formatted_cameras)} camera(s) to server")
                    
                except Exception as e:
                    ws.send(json.dumps({
                        'type': 'camera_discovery_response',
                        'requestId': request_id,
                        'success': False,
                        'error': str(e)
                    }))
                    print(f"[CAMERA DISCOVERY] Error: {e}")
            
            # Run in background thread
            threading.Thread(target=run_discovery, daemon=True).start()
        
        elif msg_type == 'firmware_update_request':
            request_id = data.get('requestId')
            update_data = data.get('update', {})
            print(f"[FIRMWARE] Update request (ID: {request_id})")
            handle_firmware_update(ws, request_id, update_data)
        
        elif msg_type == 'config_update_request':
            request_id = data.get('requestId')
            config_data = data.get('config', {})
            print(f"[CONFIG] Update request (ID: {request_id})")
            handle_config_update(ws, request_id, config_data)
        
        elif msg_type == 'restart_request':
            request_id = data.get('requestId')
            print(f"[RESTART] Request (ID: {request_id})")
            handle_restart_request(ws, request_id)
        
        elif msg_type == 'system_info_request':
            request_id = data.get('requestId')
            print(f"[SYSTEM INFO] Request (ID: {request_id})")
            info = get_system_info()
            ws.send(json.dumps({
                'type': 'system_info_response',
                'requestId': request_id,
                'info': info
            }))

    except json.JSONDecodeError:
        print(f"Invalid WebSocket message: {message}")


def on_ws_error(ws, error):
    """Handle WebSocket errors"""
    print(f"[WEBSOCKET ERROR] {error}")


def on_ws_close(ws, close_status_code, close_msg):
    """Handle WebSocket disconnection"""
    global ws_connected, should_reconnect
    ws_connected = False
    print("[WEBSOCKET] Disconnected")
    update_device_status('offline')
    
    # Trigger reconnection
    if should_reconnect:
        print("[WEBSOCKET] Will attempt to reconnect in 5 seconds...")
        time.sleep(5)
        reconnect_websocket()


def on_ws_open(ws):
    """Handle WebSocket connection"""
    global ws_connected
    ws_connected = True
    print("[WEBSOCKET] Connected successfully")
    update_device_status('online')

    # Start heartbeat with health metrics
    def send_heartbeat():
        while ws_connected:
            try:
                # Get system info and send with heartbeat
                health_metrics = get_system_info()
                ws.send(json.dumps({
                    "type": "heartbeat",
                    "healthMetrics": health_metrics
                }))
                time.sleep(15)  # Send heartbeat every 15 seconds
            except:
                break

    heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
    heartbeat_thread.start()


def connect_websocket():
    """Connect to WebSocket server with API key in URL query parameter"""
    global ws
    
    # Build WebSocket URL with API key as query parameter
    ws_url = API_URL.replace('http://', 'ws://').replace('https://', 'wss://')
    ws_url = f"{ws_url}/ws/devices?apiKey={API_KEY}"
    
    print(f"Connecting to WebSocket: {ws_url[:60]}...")
    
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_ws_message,
        on_error=on_ws_error,
        on_close=on_ws_close,
        on_open=on_ws_open
    )
    
    # Run WebSocket in separate thread
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()


def reconnect_websocket():
    """Attempt to reconnect WebSocket with exponential backoff"""
    global should_reconnect
    max_retries = 10
    retry_delay = 5  # Start with 5 seconds
    
    for attempt in range(1, max_retries + 1):
        if not should_reconnect:
            print("[WEBSOCKET] Reconnection cancelled")
            return
        
        print(f"[WEBSOCKET] Reconnection attempt {attempt}/{max_retries}")
        try:
            connect_websocket()
            # Wait a bit to see if connection succeeds
            time.sleep(3)
            if ws_connected:
                print("[WEBSOCKET] Reconnected successfully!")
                return
        except Exception as e:
            print(f"[WEBSOCKET] Reconnection failed: {e}")
        
        # Exponential backoff (max 60 seconds)
        retry_delay = min(retry_delay * 2, 60)
        if attempt < max_retries:
            print(f"[WEBSOCKET] Waiting {retry_delay} seconds before next attempt...")
            time.sleep(retry_delay)
    
    print("[WEBSOCKET] Max reconnection attempts reached. Will keep trying...")
    # Keep trying indefinitely with 60 second delay
    while should_reconnect and not ws_connected:
        time.sleep(60)
        print("[WEBSOCKET] Attempting reconnection...")
        try:
            connect_websocket()
            time.sleep(3)
            if ws_connected:
                print("[WEBSOCKET] Reconnected successfully!")
                return
        except Exception as e:
            print(f"[WEBSOCKET] Reconnection failed: {e}")


def main():
    """Main client loop"""
    print("=" * 60)
    print("StreamChanger Pi Client v" + CLIENT_VERSION)
    print("=" * 60)
    
    # Initial authentication
    if not authenticate_device():
        print("Failed to authenticate. Check API_KEY in ~/.streamchanger_config")
        sys.exit(1)
    
    # Connect WebSocket for real-time updates
    connect_websocket()
    
    print("\n[RUNNING] Client is running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Periodic sync (fallback if WebSocket misses updates)
            sync_streams()
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        global should_reconnect
        print("\n\n[SHUTDOWN] Shutting down...")
        
        # Stop reconnection attempts
        should_reconnect = False
        
        # Stop all active streams
        for stream_id in list(active_streams.keys()):
            stop_stream(stream_id)
        
        update_device_status('offline')
        print("[SUCCESS] Client stopped")


if __name__ == "__main__":
    main()