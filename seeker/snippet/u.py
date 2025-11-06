#date: 2025-11-06T17:13:49Z
#url: https://api.github.com/gists/a241995cb625974a6be136e73180c348
#owner: https://api.github.com/users/harshsoni-harsh

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from bluez_peripheral.advert import Advertisement, AdvertisingIncludes, PacketType
from bluez_peripheral.util import Adapter, get_message_bus
from dbus_next.constants import MessageType
from dbus_next.errors import DBusError, InterfaceNotFoundError
from dbus_next.message import Message
from dbus_next.signature import Variant

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
KEYS_DIR = ROOT_DIR / "keys"
SEED_PATH = KEYS_DIR / "guest_seed.bin"

LOCK_ID = "lock_01"
MQTT_BROKER = "10.0.12.142"
MQTT_PORT = 1883

MANUFACTURER_ID = 0xFFFF
ADVERT_INTERVAL = 30
ADVERT_TIMEOUT = 0
BEACON_TOKEN_SIZE = "**********"

session_key_data: Optional[Dict[str, Any]] = None


def load_seed() -> bytes:
	if not SEED_PATH.exists():
		raise RuntimeError(f"Seed file missing at {SEED_PATH}")
	seed = SEED_PATH.read_bytes()
	if len(seed) < 16:
		raise RuntimeError("Seed must be at least 16 bytes")
	return seed


def derive_beacon_token(seed: "**********":
	return hmac.new(seed, b"guest-beacon", hashlib.sha256).digest()[: "**********"


def derive_phone_id(seed: bytes) -> str:
	identifier = hashlib.sha256(seed).digest()[:8]
	return base64.urlsafe_b64encode(identifier).decode().rstrip("=")


def generate_session_token(session_key: "**********": Optional[str]) -> bytes:
	slot = int(time.time() // ADVERT_INTERVAL)
	message = (nonce or "").encode() + str(slot).encode()
	return hmac.new(session_key, message, hashlib.sha256).digest()[:16]


def on_message(client, userdata, msg):
	global session_key_data
	logger.info("Received session key on %s", msg.topic)
	session_key_data = json.loads(msg.payload.decode())


def _canonical_request_payload(lock_id: "**********": bytes, phone_id: Optional[str]) -> Dict[str, Any]:
	payload: Dict[str, Any] = {
		"lock_id": lock_id,
		"token": "**********"
		"curr_time": int(time.time()),
	}
	if phone_id:
		payload["phone_id"] = phone_id
	return payload


def get_session_key(lock_id: "**********": bytes, phone_id: Optional[str]) -> tuple[bytes, float, Optional[str]]:
	global session_key_data
	session_key_data = None
	client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
	client.on_message = on_message

	try:
		client.connect(MQTT_BROKER, MQTT_PORT, 60)
	except OSError as exc:
		raise ConnectionError(
			f"Failed to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}: {exc}"
		) from exc

	guest_topic = f"guests/{lock_id}/session"
	request_topic = "backend/session_requests"
	request_payload = "**********"

	client.loop_start()
	try:
		client.subscribe(guest_topic, qos=1)
		publish_result = client.publish(request_topic, request_payload, qos=1)
		publish_result.wait_for_publish()
		logger.info("Requested session key for %s", lock_id)

		deadline = time.time() + 15
		while session_key_data is None and time.time() < deadline:
			time.sleep(0.1)
	finally:
		client.loop_stop()
		client.disconnect()

	if not session_key_data:
		raise TimeoutError("No session key received from backend within timeout")

	session_key = base64.b64decode(session_key_data["session_key"])
	expiry = float(session_key_data.get("expiry", time.time() + 300))
	nonce = session_key_data.get("nonce")
	logger.info("Session key received. Expires: %s, nonce: %s", expiry, nonce)
	return session_key, expiry, nonce


async def get_first_adapter(bus) -> Adapter:
	message = Message(
		destination="org.bluez",
		path="/",
		interface="org.freedesktop.DBus.ObjectManager",
		member="GetManagedObjects",
	)
	reply = await bus.call(message)
	if reply.message_type == MessageType.ERROR:
		raise RuntimeError(
			f"GetManagedObjects failed: {reply.error_name} {reply.body}"
		)

	objects: Dict[str, Dict[str, Any]] = reply.body[0]
	for path, interfaces in objects.items():
		if "org.bluez.Adapter1" in interfaces:
			introspection = await bus.introspect("org.bluez", path)
			proxy = bus.get_proxy_object("org.bluez", path, introspection)
			return Adapter(proxy)

	raise ValueError("No bluetooth adapters could be found.")


def _ensure_le_advertising_support(adapter: Adapter) -> None:
	try:
		adapter._proxy.get_interface("org.bluez.LEAdvertisingManager1")
	except InterfaceNotFoundError as exc:
		raise RuntimeError(
			"Adapter does not expose LEAdvertisingManager1; start bluetoothd with --experimental or use advertising-capable hardware."
		) from exc


async def _set_adapter_property(adapter: Adapter, name: str, value: bool) -> None:
	try:
		props = adapter._proxy.get_interface("org.freedesktop.DBus.Properties")
		await props.call_set("org.bluez.Adapter1", name, Variant("b", value))  # type: ignore[attr-defined]
	except (InterfaceNotFoundError, AttributeError):
		return
	except DBusError as exc:
		err_name = getattr(exc, "name", "")
		if err_name not in {
			"org.freedesktop.DBus.Error.PropertyReadOnly",
			"org.bluez.Error.NotPermitted",
			"org.bluez.Error.NotSupported",
		}:
			logger.debug("Unable to set adapter property %s: %s", name, exc)


async def _stop_discovery(adapter: Adapter) -> None:
	try:
		await adapter.stop_discovery()  # type: ignore[attr-defined]
		return
	except AttributeError:
		pass
	except DBusError:
		return
	try:
		iface = adapter._proxy.get_interface("org.bluez.Adapter1")
		await iface.call_stop_discovery()  # type: ignore[attr-defined]
	except Exception:
		pass


async def _prepare_adapter_for_advertising(adapter: Adapter) -> None:
	_ensure_le_advertising_support(adapter)
	await _set_adapter_property(adapter, "Powered", True)
	await _stop_discovery(adapter)


def _describe_advertising_error(exc: Exception, adapter_name: str) -> str:
	if isinstance(exc, RuntimeError):
		return str(exc)
	if isinstance(exc, DBusError):
		err_name = getattr(exc, "name", "")
		if err_name == "org.bluez.Error.NotSupported":
			return (
				f"Adapter {adapter_name} rejected LE advertising (NotSupported); ensure bluetoothd is running with --experimental and the adapter firmware allows advertising."
			)
		if err_name == "org.bluez.Error.AlreadyExists":
			return (
				"An advertisement with the same path is still registered. Restart bluetoothd or wait for the previous session to expire."
			)
		if err_name == "org.bluez.Error.NotAuthorized":
			return (
				"DBus reported NotAuthorized; run under a user in the bluetooth group or adjust policy to allow advertising."
			)
		return f"{err_name or 'DBusError'}: {exc}"
	return str(exc)


class BeaconAdvertisement(Advertisement):
	def __init__(self, token: "**********":
		self.token = "**********"
		super().__init__(
			localName="",
			serviceUUIDs=[],
			appearance=0x0000,
			timeout=ADVERT_TIMEOUT,
			discoverable=False,
			packet_type=PacketType.BROADCAST,
			manufacturerData={MANUFACTURER_ID: "**********"
			includes=AdvertisingIncludes.NONE,
		)
		self._manufacturerData[MANUFACTURER_ID] = "**********"
		self._advert_path = f"/com/ble_lock/unlocker/beacon_{uuid.uuid4().hex}"

	async def start(self, bus, adapter):
		await super().register(bus, adapter, self._advert_path)

	async def stop(self, adapter):
		interface = adapter._proxy.get_interface(self._MANAGER_INTERFACE)
		with suppress(DBusError):
			await interface.call_unregister_advertisement(self._advert_path)


class SessionAdvertisement(Advertisement):
	def __init__(self, session_key: bytes, nonce: Optional[str]):
		self.session_key = session_key
		self.nonce = nonce
		self.current_token = "**********"
		super().__init__(
			localName="",
			serviceUUIDs=[],
			appearance=0x0000,
			timeout=ADVERT_TIMEOUT,
			discoverable=False,
			packet_type=PacketType.BROADCAST,
			manufacturerData={MANUFACTURER_ID: "**********"
			includes=AdvertisingIncludes.NONE,
		)
		self._manufacturerData[MANUFACTURER_ID] = "**********"
		self._advert_path = f"/com/ble_lock/unlocker/session_{uuid.uuid4().hex}"

	async def start(self, bus, adapter):
		await super().register(bus, adapter, self._advert_path)

	async def stop(self, adapter):
		interface = adapter._proxy.get_interface(self._MANAGER_INTERFACE)
		with suppress(DBusError):
			await interface.call_unregister_advertisement(self._advert_path)

	def refresh(self):
		self.current_token = "**********"
		self._manufacturerData[MANUFACTURER_ID] = "**********"
		logger.debug("Refreshed session token: "**********"

	@property
	def counter(self) -> int:
		return int(time.time() // ADVERT_INTERVAL)


async def advertise_beacon(token: "**********": Optional[asyncio.Future] = None) -> None:
	bus = await get_message_bus()
	adapter: Optional[Adapter] = None
	advert: Optional[BeaconAdvertisement] = None
	adapter_name = "<unknown>"
	try:
		adapter = await get_first_adapter(bus)
		adapter_name = await adapter.get_name()
		await _prepare_adapter_for_advertising(adapter)
		advert = "**********"
		await advert.start(bus, adapter)
		logger.info("Beacon advertising started on adapter %s", adapter_name)
		if ready_future and not ready_future.done():
			ready_future.set_result(True)
		while True:
			await asyncio.sleep(5)
	except asyncio.CancelledError:
		pass
	except Exception as exc:
		logger.error("Failed to advertise beacon: %s", _describe_advertising_error(exc, adapter_name))
		if ready_future and not ready_future.done():
			ready_future.set_exception(exc)
	finally:
		with suppress(Exception):
			if advert and adapter:
				await advert.stop(adapter)
		with suppress(Exception):
			bus.disconnect()
		logger.info("Beacon advertising stopped")


async def advertise_session(
	session_key: bytes,
	expiry: float,
	nonce: Optional[str],
	ready_future: Optional[asyncio.Future] = None,
) -> None:
	bus = await get_message_bus()
	adapter: Optional[Adapter] = None
	advert: Optional[SessionAdvertisement] = None
	adapter_name = "<unknown>"
	try:
		adapter = await get_first_adapter(bus)
		adapter_name = await adapter.get_name()
		await _prepare_adapter_for_advertising(adapter)
		advert = SessionAdvertisement(session_key, nonce)
		await advert.start(bus, adapter)
		logger.info("Session advertising started on adapter %s", adapter_name)
		if ready_future and not ready_future.done():
			ready_future.set_result(advert)
		while True:
			remaining = expiry - time.time()
			if remaining <= 0:
				logger.info("Session key expired; stopping advertising")
				break
			await asyncio.sleep(min(ADVERT_INTERVAL, max(0.5, remaining)))
			advert.refresh()
			await advert.stop(adapter)
			await advert.start(bus, adapter)
	except asyncio.CancelledError:
		pass
	except Exception as exc:
		logger.error("Failed to advertise session: %s", _describe_advertising_error(exc, adapter_name))
		if ready_future and not ready_future.done():
			ready_future.set_exception(exc)
	finally:
		with suppress(Exception):
			if advert and adapter:
				await advert.stop(adapter)
		with suppress(Exception):
			bus.disconnect()
		logger.info("Session advertising stopped")


async def main() -> None:
	seed = load_seed()
	beacon_token = "**********"
	phone_id = derive_phone_id(seed)

	beacon_ready: asyncio.Future = asyncio.get_running_loop().create_future()
	beacon_task = "**********"=beacon_ready))
	try:
		await beacon_ready
		logger.info("Requesting session key from backend...")
		session_key, expiry, nonce = await asyncio.to_thread(
			get_session_key,
			LOCK_ID,
			beacon_token,
			phone_id,
		)
	finally:
		beacon_task.cancel()
		with suppress(Exception):
			await beacon_task

	session_ready: asyncio.Future = asyncio.get_running_loop().create_future()
	session_task = asyncio.create_task(
		advertise_session(session_key, expiry, nonce, ready_future=session_ready)
	)
	try:
		await session_task
	except asyncio.CancelledError:
		pass


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		logger.info("Script stopped by user")
	except Exception as exc:
		logger.exception("Guest unlocker encountered an error: %s", exc)
:
		await session_task
	except asyncio.CancelledError:
		pass


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		logger.info("Script stopped by user")
	except Exception as exc:
		logger.exception("Guest unlocker encountered an error: %s", exc)
