#date: 2025-02-05T17:06:17Z
#url: https://api.github.com/gists/3bbb51566a7ba1a7bbc1a7e2d384b8b6
#owner: https://api.github.com/users/gary23w

import argparse
import logging
import logging.handlers
import re
import requests
import ipaddress
import time
import threading
from collections import defaultdict
from pwn import hexdump
import pickle
import numpy as np
from scapy.all import sniff, Ether, IP, IPv6, TCP, UDP, ICMP, ARP, Raw

try:
    from scapy.layers.tls.all import TLS, TLSClientHello, TLSHandshake
except ImportError:
    TLS = None
    TLSClientHello = None
    TLSHandshake = None

class NoColorFormatter(logging.Formatter):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    def format(self, record):
        original = super().format(record)
        return self.ansi_escape.sub('', original)

ip_check_cache = {}
ip_cache_lock = threading.Lock()
CACHE_TTL = 3600  

BACKUPS = 2  
FLAGGED_LOG_BACKUPS = 2  

COMMON_PORTS = {
    20: ("FTP Data", "File Transfer Protocol - Data channel"),
    21: ("FTP Control", "File Transfer Protocol - Control channel"),
    22: ("SSH", "Secure Shell"),
    23: ("Telnet", "Telnet protocol"),
    25: ("SMTP", "Simple Mail Transfer Protocol"),
    53: ("DNS", "Domain Name System"),
    67: ("DHCP", "Dynamic Host Configuration Protocol (Server)"),
    68: ("DHCP", "Dynamic Host Configuration Protocol (Client)"),
    80: ("HTTP", "Hypertext Transfer Protocol"),
    110: ("POP3", "Post Office Protocol v3"),
    119: ("NNTP", "Network News Transfer Protocol"),
    123: ("NTP", "Network Time Protocol"),
    143: ("IMAP", "Internet Message Access Protocol"),
    161: ("SNMP", "Simple Network Management Protocol"),
    194: ("IRC", "Internet Relay Chat"),
    443: ("HTTPS", "HTTP Secure"),
    465: ("SMTPS", "SMTP Secure (over SSL)"),
    993: ("IMAPS", "IMAP Secure (over SSL)"),
    995: ("POP3S", "POP3 Secure (over SSL)"),
    3306: ("MySQL", "MySQL database service"),
    5432: ("PostgreSQL", "PostgreSQL database service"),
    3389: ("RDP", "Remote Desktop Protocol")
}

def setup_flagged_logger(log_file: str, log_level: str = "INFO") -> logging.Logger:
    """
    Creates a dedicated logger for flagged IP events that rotates the log after two backups.
    """
    flagged_logger = logging.getLogger("FlaggedIPLogger")
    flagged_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = NoColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
    if not flagged_logger.handlers:
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=2*1024*1024, backupCount=FLAGGED_LOG_BACKUPS
        )
        fh.setFormatter(formatter)
        flagged_logger.addHandler(fh)
    return flagged_logger

flagged_ip_logger = None

def analyze_hex_dump(payload: bytes, logger: logging.Logger):
    """
    Analyze the hex dump of the payload for known tell-tale signatures.
    Checks the first few bytes (and optionally the entire payload) for known
    file signatures or flag markers. Logs a warning if any are detected.
    """
    head_hex = payload[:16].hex().upper()
    full_hex = payload.hex().upper()

    logger.info("Checking payload hex dump for known signatures...")
    SIGNATURES = {
        "504B0304": "ZIP archive / Office Open XML document (DOCX, XLSX, PPTX)",
        "504B030414000600": "Office Open XML (DOCX, XLSX, PPTX) extended header",
        "1F8B08": "GZIP archive",
        "377ABCAF271C": "7-Zip archive",
        "52617221": "RAR archive",
        "425A68": "BZIP2 archive",
        "213C617263683E0A": "Ar (UNIX archive) / Debian package",
        "7F454C46": "ELF executable (Unix/Linux)",
        "4D5A": "Windows executable (EXE, MZ header / DLL)",
        "CAFEBABE": "Java class file or Mach-O Fat Binary (ambiguous)",
        "FEEDFACE": "Mach-O executable (32-bit, little-endian)",
        "CEFAEDFE": "Mach-O executable (32-bit, big-endian)",
        "FEEDFACF": "Mach-O executable (64-bit, little-endian)",
        "CFFAEDFE": "Mach-O executable (64-bit, big-endian)",
        "BEBAFECA": "Mach-O Fat Binary (little endian)",
        "4C000000": "Windows shortcut file (.lnk)",
        "4D534346": "Microsoft Cabinet file (CAB)",
        "D0CF11E0": "Microsoft Office legacy format (DOC, XLS, PPT)",
        "25504446": "PDF document",
        "7B5C727466": "RTF document (starting with '{\\rtf')",
        "3C3F786D6C": "XML file (<?xml)",
        "3C68746D6C3E": "HTML file",
        "252150532D41646F6265": "PostScript/EPS document (starts with '%!PS-Adobe')",
        "4D2D2D2D": "PostScript file (---)",
        "89504E47": "PNG image",
        "47494638": "GIF image",
        "FFD8FF": "JPEG image (general)",
        "FFD8FFE0": "JPEG image (JFIF)",
        "FFD8FFE1": "JPEG image (EXIF)",
        "424D": "Bitmap image (BMP)",
        "49492A00": "TIFF image (little endian / Intel)",
        "4D4D002A": "TIFF image (big endian / Motorola)",
        "38425053": "Adobe Photoshop document (PSD)",
        "00000100": "ICO icon file",
        "00000200": "CUR cursor file",
        "494433": "MP3 audio (ID3 tag)",
        "000001BA": "MPEG video (VCD)",
        "000001B3": "MPEG video",
        "66747970": "MP4/MOV file (ftyp)",
        "4D546864": "MIDI file",
        "464F524D": "AIFF audio file",
        "52494646": "AVI file (RIFF) [Also used in WAV]",
        "664C6143": "FLAC audio file",
        "4F676753": "OGG container file (OggS)",
        "53514C69": "SQLite database file (SQLite format 3)",
        "420D0D0A": "Python compiled file (.pyc) [example magic, may vary]",
        "6465780A": "Android Dalvik Executable (DEX) file",
        "EDABEEDB": "RPM package file",
        "786172210D0A1A0A": "XAR archive (macOS installer package)",
    }

    found = []
    for sig, desc in SIGNATURES.items():
        if head_hex.startswith(sig) or sig in full_hex:
            found.append((sig, desc))
    if found:
        for sig, desc in found:
            logger.warning("RED ALERT: Detected signature %s (%s) in payload.", sig, desc)
            logger.warning("Full hex dump: %s", full_hex)
    return found

# ML future implementation
def classify_payload_ml(payload: bytes, logger: logging.Logger) -> str:
    """
    Example ML-based payload classification.
    """
    try:
        features = np.frombuffer(payload[:64], dtype=np.uint8)
        if features.shape[0] < 64:
            features = np.pad(features, (0, 64 - features.shape[0]), mode='constant')
        features = features.astype(np.float32) / 255.0
        features = features.reshape(1, -1)
        with open("payload_classifier.pkl", "rb") as f:
            model = pickle.load(f)
        prediction = model.predict(features)
        logger.info("ML Classifier Prediction: %s", prediction[0])
        return prediction[0]
    except Exception as e:
        logger.error("Error classifying payload with ML model: %s", e)
        return "Unknown"

def setup_logger(log_file: str, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("PacketSniffer")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = NoColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=2*1024*1024, backupCount=BACKUPS)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def format_ip_details(ip_address: str, info: dict) -> str:
    lines = [
        "[IP Background Check]",
        f"  IP: {ip_address}"
    ]
    for key in ("hostname", "city", "region", "country", "org"):
        if key in info:
            lines.append(f"  {key.capitalize()}: {info[key]}")
    return "\n".join(lines)

def get_ip_details(ip_address: str, logger: logging.Logger) -> str:
    info = get_ip_info(ip_address, logger)
    if info:
        return format_ip_details(ip_address, info)
    return ""

def get_ip_info(ip_address: str, logger: logging.Logger) -> dict:
    try:
        ip_obj = ipaddress.ip_address(ip_address)
        if ip_obj.is_loopback or ip_obj.is_link_local:
            return {}
    except Exception as e:
        logger.error(f"Error checking IP properties for {ip_address}: {e}")
        return {}
    current_time = time.time()
    with ip_cache_lock:
        if ip_address in ip_check_cache:
            cached_time, info = ip_check_cache[ip_address]
            if current_time - cached_time < CACHE_TTL:
                return info
            else:
                del ip_check_cache[ip_address]
    try:
        response = requests.get(f"https://ipinfo.io/{ip_address}/json", timeout=5)
        if response.status_code == 200:
            info = response.json()
            with ip_cache_lock:
                ip_check_cache[ip_address] = (current_time, info)
            return info
        else:
            logger.warning(f"[IP Background Check] Unable to retrieve details for {ip_address}. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"[IP Background Check] Error checking {ip_address}: {e}")
    return {}

def identify_application(src_port: int, dst_port: int):
    for port in (dst_port, src_port):
        if port in COMMON_PORTS:
            return COMMON_PORTS[port]
    return ("Unknown", "Unknown application protocol")

def parse_http_payload(payload: bytes, logger: logging.Logger):
    try:
        text = payload.decode('utf-8', errors='replace')
        logger.info("HTTP Payload (parsed):")
        for line in text.splitlines():
            logger.info("  " + line)
    except Exception as e:
        logger.error("Error parsing HTTP payload: " + str(e))

def parse_dns_payload(payload: bytes, logger: logging.Logger):
    if len(payload) < 12:
        logger.warning("DNS payload too short to parse.")
        return
    try:
        import struct
        transaction_id, flags, qdcount, ancount, nscount, arcount = struct.unpack("!HHHHHH", payload[:12])
        logger.info("DNS Header:")
        logger.info(f"  Transaction ID: {transaction_id:#04x}")
        logger.info(f"  Flags         : {flags:#04x}")
        logger.info(f"  Questions     : {qdcount}")
        logger.info(f"  Answers       : {ancount}")
        logger.info(f"  Authority RRs : {nscount}")
        logger.info(f"  Additional RRs: {arcount}")
    except Exception as e:
        logger.error("Error parsing DNS payload: " + str(e))

def parse_text_payload(payload: bytes, logger: logging.Logger):
    try:
        text = payload.decode('utf-8', errors='replace')
        logger.info("Text Payload:")
        logger.info(text)
    except Exception as e:
        logger.error("Error decoding payload as text: " + str(e))

def parse_tls_payload(payload: bytes, logger: logging.Logger):
    logger.info("Raw TLS Payload (hex dump):")
    logger.info("\n%s", hexdump(payload))

def get_local_process_info(port: int):
    """
    Attempt to find the process ID associated with a given local port.
    This uses psutil and works only if the port is being used locally.
    """
    try:
        import psutil
    except ImportError:
        return "N/A (psutil not installed)"
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == port:
            return conn.pid if conn.pid is not None else "N/A"
    return "N/A"

def log_flagged_ip(packet, flagged_signatures, app_name, app_details):
    """
    Log detailed information about the flagged IP to the separate flagged IP log file.
    """
    if packet.haslayer(IP):
        ip_layer = packet[IP]
        source_ip = ip_layer.src
        dest_ip = ip_layer.dst
    elif packet.haslayer(IPv6):
        ip_layer = packet[IPv6]
        source_ip = ip_layer.src
        dest_ip = ip_layer.dst
    else:
        source_ip = "Unknown"
        dest_ip = "Unknown"

    port_info = ""
    process_id = "N/A"
    if packet.haslayer(TCP):
        tcp_layer = packet[TCP]
        port_info = f"TCP src: {tcp_layer.sport}, dst: {tcp_layer.dport}"
        process_id = get_local_process_info(tcp_layer.dport)
    elif packet.haslayer(UDP):
        udp_layer = packet[UDP]
        port_info = f"UDP src: {udp_layer.sport}, dst: {udp_layer.dport}"
        process_id = get_local_process_info(udp_layer.dport)
    
    ip_background = get_ip_details(source_ip, logging.getLogger("PacketSniffer"))
    if not ip_background:
        ip_background = "No background info available."

    message = (
        "\n====== FLAGGED IP ALERT ======\n"
        f"Source IP: {source_ip}\n"
        f"Destination IP: {dest_ip}\n"
        f"Application: {app_name} ({app_details})\n"
        f"Port Info: {port_info}\n"
        f"Process ID: {process_id}\n"
        f"IP Background:\n{ip_background}\n"
        f"Flagged Signatures: {flagged_signatures}\n"
        "===============================\n"
    )
    if flagged_ip_logger:
        flagged_ip_logger.warning(message)
    else:
        print("Flagged IP Logger not configured.")

def parse_payload(packet, app_name: str, payload: bytes, logger: logging.Logger):
    """
    Parses the payload and, if a signature is flagged, logs detailed IP info
    to the separate flagged IP log file.
    """
    logger.info("Parsing payload for application: %s", app_name)
    flagged_signatures = analyze_hex_dump(payload, logger)
    if flagged_signatures:
        # Log flagged IP details if signatures are found.
        if packet.haslayer(TCP):
            ports = (packet[TCP].sport, packet[TCP].dport)
        elif packet.haslayer(UDP):
            ports = (packet[UDP].sport, packet[UDP].dport)
        else:
            ports = (0, 0)
        app_info = identify_application(*ports)
        log_flagged_ip(packet, flagged_signatures, app_name, app_info[1])
    
    if "HTTP" in app_name:
        parse_http_payload(payload, logger)
    elif app_name == "DNS":
        parse_dns_payload(payload, logger)
    else:
        parse_text_payload(payload, logger)

def packet_handler(packet):
    global args, logger
    if packet.haslayer(IP):
        ip_str = packet[IP].src
        ip_obj = ipaddress.ip_address(ip_str)
        if args.local_only:
            if not ip_obj.is_private:
                return
        else:
            if ip_obj.is_private:
                return
            if args.red_alert:
                ip_info = get_ip_info(ip_str, logger)
                if ip_info.get("country", "").upper() in args.whitelist:
                    return
    elif packet.haslayer(IPv6):
        ip_str = packet[IPv6].src
        ip_obj = ipaddress.ip_address(ip_str)
        if args.local_only:
            if not ip_obj.is_private:
                return
        else:
            if ip_obj.is_private:
                return
            if args.red_alert:
                ip_info = get_ip_info(ip_str, logger)
                if ip_info.get("country", "").upper() in args.whitelist:
                    return

    logger.info("=" * 80)
    logger.info("Packet: %s", packet.summary())

    if packet.haslayer(Ether):
        eth = packet[Ether]
        logger.info("Ethernet: src=%s, dst=%s, type=0x%04x", eth.src, eth.dst, eth.type)
    else:
        logger.warning("No Ethernet layer found.")
        return

    if packet.haslayer(ARP):
        arp = packet[ARP]
        logger.info("ARP: op=%s, src=%s, dst=%s", arp.op, arp.psrc, arp.pdst)
        return

    if packet.haslayer(IP):
        ip_layer = packet[IP]
        logger.info("IPv4: src=%s, dst=%s, ttl=%s, proto=%s", ip_layer.src, ip_layer.dst, ip_layer.ttl, ip_layer.proto)
        if not args.no_bgcheck:
            details = get_ip_details(ip_layer.src, logger)
            if details:
                for line in details.splitlines():
                    logger.info(line)
    elif packet.haslayer(IPv6):
        ip_layer = packet[IPv6]
        logger.info("IPv6: src=%s, dst=%s, hlim=%s", ip_layer.src, ip_layer.dst, ip_layer.hlim)
        if not args.no_bgcheck:
            details = get_ip_details(ip_layer.src, logger)
            if details:
                logger.info(details)
    else:
        logger.warning("No IP/IPv6 layer found.")
        return

    if packet.haslayer(TCP):
        tcp_layer = packet[TCP]
        logger.info("TCP: sport=%s, dport=%s", tcp_layer.sport, tcp_layer.dport)
        app_name, app_details = identify_application(tcp_layer.sport, tcp_layer.dport)
        logger.info("Identified Application: %s (%s)", app_name, app_details)
        if app_name == "HTTPS" or (TLS and packet.haslayer(TLS)):
            if TLS and packet.haslayer(TLS):
                tls_layer = packet[TLS]
                logger.info("TLS Record: %s", tls_layer.summary())
                if packet.haslayer(TLSClientHello):
                    client_hello = packet[TLSClientHello]
                    logger.info("TLS ClientHello: %s", client_hello.summary())
                    if hasattr(client_hello, 'servernames'):
                        logger.info("SNI: %s", client_hello.servernames)
            else:
                if packet.haslayer(Raw):
                    payload = bytes(packet[Raw].load)
                    parse_tls_payload(payload, logger)
        else:
            if packet.haslayer(Raw):
                payload = bytes(packet[Raw].load)
                parse_payload(packet, app_name, payload, logger)
    elif packet.haslayer(UDP):
        udp_layer = packet[UDP]
        logger.info("UDP: sport=%s, dport=%s", udp_layer.sport, udp_layer.dport)
        app_name, app_details = identify_application(udp_layer.sport, udp_layer.dport)
        logger.info("Identified Application: %s (%s)", app_name, app_details)
        if packet.haslayer(Raw):
            payload = bytes(packet[Raw].load)
            parse_payload(packet, app_name, payload, logger)
    elif packet.haslayer(ICMP):
        icmp_layer = packet[ICMP]
        logger.info("ICMP: type=%s, code=%s", icmp_layer.type, icmp_layer.code)
    else:
        logger.warning("Unsupported transport layer.")

def main():
    global args, logger, flagged_ip_logger
    parser = argparse.ArgumentParser(
        description="Enhanced Packet Sniffer with Scapy, TLS Parsing, and Filtering Options"
    )
    parser.add_argument("-i", "--interface", type=str, default="eth0",
                        help="Network interface to sniff on (default: eth0)")
    parser.add_argument("-l", "--logfile", type=str, default="sniffer.log",
                        help="Path to the log file (default: sniffer.log)")
    parser.add_argument("--no-bgcheck", action="store_true",
                        help="Disable IP lookup (background check)")
    parser.add_argument("--red-alert", action="store_true",
                        help="Enable red alert mode: only log packets from non-allied countries")
    parser.add_argument("--whitelist", type=str, default="US",
                        help="Comma-separated list of allied (whitelisted) country codes (default: US)")
    parser.add_argument("--local-only", action="store_true",
                        help="Enable local-only mode: show only packets from local (private) IPs and block all others")
    parser.add_argument("-v", "--verbosity", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()
    args.whitelist = set(code.strip().upper() for code in args.whitelist.split(','))

    logger = setup_logger(args.logfile, args.verbosity)
    flagged_ip_logger = setup_flagged_logger("flagged_ips.log", args.verbosity)
    
    logger.info("Starting Enhanced Packet Sniffer on interface '%s'", args.interface)

    try:
        sniff(iface=args.interface, prn=packet_handler, store=0)
    except KeyboardInterrupt:
        logger.info("Stopping packet capture (KeyboardInterrupt received)...")
    except Exception as e:
        logger.error("Error during packet capture: %s", e)

if __name__ == "__main__":
    main()
