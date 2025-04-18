#date: 2025-04-18T16:36:19Z
#url: https://api.github.com/gists/96054fef7ca1e2099cdd7295a5f9eecd
#owner: https://api.github.com/users/lzlrd

"""
Immich WebDAV Wrapper

Connects to Immich API and serves album assets via WebDAV.
"""

import hashlib
import logging
import os
import requests
import threading
import time
import json
from cheroot import wsgi
from datetime import datetime
from dateutil.parser import isoparse
from dotenv import load_dotenv
from wsgidav.wsgidav_app import WsgiDAVApp
from wsgidav.dav_provider import DAVProvider, DAVCollection, DAVNonCollection
from wsgidav.dav_error import (
    DAVError,
    HTTP_NOT_FOUND,
    HTTP_INTERNAL_ERROR,
    HTTP_FORBIDDEN,
)
from wsgidav import util
from typing import List, Dict, Optional, Union, Generator, Set


DEFAULT_REFRESH_HOURS = 1
DEFAULT_WEBDAV_PORT = 1700
DEFAULT_FLATTEN = False


_logger = util.get_module_logger(__name__)

_logger.setLevel(logging.INFO)


def get_file_extension(filename: str) -> str:
    """Safely get the file extension in lowercase."""
    if not isinstance(filename, str) or "." not in filename or filename.startswith("."):
        return ""
    return filename.split(".")[-1].lower()


def safe_isoparse_timestamp(date_str: Optional[str]) -> Optional[int]:
    """Parse ISO date string to int timestamp, handling errors."""
    if not date_str:
        return 0
    try:
        return int(isoparse(date_str).timestamp())
    except (ValueError, TypeError, OverflowError) as e:
        _logger.warning(f"Error parsing date string '{date_str}': {e}")
        return 0


class ImmichAsset(DAVNonCollection):
    """Represents a single Immich asset (photo/video) as a DAV resource (file)."""

    def __init__(self, path: str, environ: dict, asset_data: Dict):
        if not isinstance(path, str) or not path.startswith("/"):
            _logger.error(f"ImmichAsset initialized with invalid path: {path!r}")
            raise ValueError(f"Invalid path for ImmichAsset: {path!r}")
        if not isinstance(asset_data, dict):
            _logger.error(
                f"ImmichAsset initialized with invalid asset_data for path {path}"
            )
            raise ValueError("asset_data must be a dictionary")
        super().__init__(path, environ)
        self.asset = asset_data
        self.asset_id = asset_data.get("id", "N/A")
        _logger.debug(f"ImmichAsset created: {self.path} (ID: {self.asset_id})")

    def _get_fs_path(self) -> Optional[str]:
        """Get the filesystem path, handling potential mounting differences."""
        original_path = self.asset.get("originalPath")
        if not original_path:
            _logger.warning(
                f"Missing 'originalPath' for asset ID {self.asset_id} ({self.path})"
            )
            return None

        fs_path = (
            original_path if original_path.startswith("/") else f"/{original_path}"
        )
        return fs_path

    def get_content_length(self) -> Optional[int]:
        fs_path = self._get_fs_path()
        if not fs_path:
            return 0

        try:
            size = os.path.getsize(fs_path)
            _logger.debug(
                f"get_content_length for {self.path}: found path {fs_path}, size {size}"
            )
            return size
        except FileNotFoundError:
            _logger.error(
                f"FileNotFoundError getting size for asset ID {self.asset_id}: Path '{fs_path}' not accessible."
            )
            return 0
        except OSError as e:
            _logger.error(
                f"OSError getting size for {fs_path} (asset ID {self.asset_id}): {e}"
            )
            return 0

    def get_content_type(self) -> Optional[str]:
        return self.asset.get("originalMimeType", "application/octet-stream")

    def get_creation_date(self) -> int:
        return safe_isoparse_timestamp(self.asset.get("fileCreatedAt"))

    def get_last_modified(self) -> int:

        return (
            safe_isoparse_timestamp(self.asset.get("fileModifiedAt"))
            or self.get_creation_date()
        )

    def get_display_name(self) -> str:

        return self.name

    def get_etag(self) -> Optional[str]:

        last_modified = self.get_last_modified()
        asset_id = self.asset.get("id")
        if not asset_id or last_modified is None:
            _logger.debug(
                f"Cannot generate ETag for {self.path} due to missing ID or mod_time"
            )
            return None

        return f"{asset_id}-{last_modified}"

    def support_etag(self) -> bool:
        return True

    def support_ranges(self) -> bool:
        return True

    def get_content(self):
        """Returns a seekable file-like object for the asset."""
        fs_path = self._get_fs_path()
        if not fs_path:
            _logger.error(
                f"Cannot get content: Missing or invalid 'originalPath' for asset ID {self.asset_id}"
            )
            raise DAVError(
                HTTP_INTERNAL_ERROR, "Asset filesystem path missing in metadata"
            )

        try:

            _logger.debug(f"Opening file for get_content: {fs_path}")
            file_obj = open(fs_path, "rb")
            return file_obj
        except FileNotFoundError:
            _logger.error(
                f"FileNotFoundError getting content for {fs_path} (asset ID {self.asset_id})"
            )
            raise DAVError(HTTP_NOT_FOUND, f"File not found: {fs_path}")
        except PermissionError as e:
            _logger.error(f"Permission error for {fs_path}: {e}")
            raise DAVError(HTTP_FORBIDDEN, f"Permission denied: {e}")
        except OSError as e:
            _logger.error(
                f"OSError getting content for {fs_path} (asset ID {self.asset_id}): {e}"
            )
            raise DAVError(HTTP_INTERNAL_ERROR, f"Error reading asset file: {e}")


class ImmichAssetCollection(DAVCollection):
    """Represents a collection of assets of a specific type (e.g., 'images', 'videos') within an album."""

    def __init__(self, path: str, environ: dict, album_data: Dict, asset_type: str):
        if not isinstance(path, str) or not path.startswith("/"):
            _logger.error(
                f"ImmichAssetCollection initialized with invalid path: {path!r}"
            )
            raise ValueError(f"Invalid path for ImmichAssetCollection: {path!r}")
        if not isinstance(album_data, dict):
            _logger.error(
                f"ImmichAssetCollection initialized with invalid album_data for path {path}"
            )
            raise ValueError("album_data must be a dictionary")
        super().__init__(path, environ)
        self.album = album_data
        self.asset_type_filter = asset_type.upper()
        self.excluded_types = getattr(self.provider, "filetype_ignore_list", set())

        self.member_assets = self._filter_assets()
        _logger.debug(
            f"ImmichAssetCollection created: {self.path} (type: {self.asset_type_filter}, count: {len(self.member_assets)})"
        )

    def _filter_assets(self) -> Dict[str, Dict]:
        """Filter assets from the album matching the type and exclusion list."""
        filtered = {}
        assets_list = self.album.get("assets", [])
        if not isinstance(assets_list, list):
            _logger.warning(
                f"Invalid 'assets' data in album {self.album.get('id')} for {self.path}"
            )
            return {}

        for asset in assets_list:
            if not isinstance(asset, dict):
                continue

            asset_type = asset.get("type")
            filename = asset.get("originalFileName")

            if asset_type == self.asset_type_filter and filename:
                ext = get_file_extension(filename)
                if ext not in self.excluded_types:
                    if filename in filtered:
                        _logger.warning(
                            f"Duplicate filename '{filename}' in album {self.album.get('id')}, type {self.asset_type_filter}. Overwriting."
                        )
                    filtered[filename] = asset
        return filtered

    def get_member_names(self) -> List[str]:
        """Return sorted list of asset filenames."""
        names = sorted(self.member_assets.keys())
        _logger.debug(f"get_member_names for {self.path} returning: {names}")
        return names

    def get_member(self, name: str) -> ImmichAsset:
        """Get a specific asset (file). Must exist if called."""
        _logger.debug(f"get_member searching for asset '{name}' in {self.path}")
        asset_data = self.member_assets.get(name)
        if asset_data:

            member_path = util.join_uri(self.path, name)
            return ImmichAsset(member_path, self.environ, asset_data)
        else:
            _logger.debug(
                f"Asset '{name}' not found in pre-filtered assets for {self.path}."
            )
            raise DAVError(HTTP_NOT_FOUND)

    def get_creation_date(self) -> Optional[int]:
        return safe_isoparse_timestamp(self.album.get("createdAt"))

    def get_last_modified(self) -> Optional[int]:
        return (
            safe_isoparse_timestamp(self.album.get("updatedAt"))
            or self.get_creation_date()
        )

    def get_display_name(self) -> str:
        return self.name


class ImmichAlbumCollection(DAVCollection):
    """Represents a single Immich album as a DAV collection (directory)."""

    def __init__(self, path: str, environ: dict, album_data: Dict):
        if not isinstance(path, str) or not path.startswith("/"):
            _logger.error(
                f"ImmichAlbumCollection initialized with invalid path: {path!r}"
            )
            raise ValueError(f"Invalid path for ImmichAlbumCollection: {path!r}")
        if not isinstance(album_data, dict):
            _logger.error(
                f"ImmichAlbumCollection initialized with invalid album_data for path {path}"
            )
            raise ValueError("album_data must be a dictionary")
        super().__init__(path, environ)
        self.album = album_data
        self.flatten = getattr(self.provider, "flatten_structure", DEFAULT_FLATTEN)
        self.excluded_types = getattr(self.provider, "filetype_ignore_list", set())

        self._members: Optional[Dict[str, Union[Dict, str]]] = None
        self._member_names: Optional[List[str]] = None

        self._initialize_members()
        _logger.debug(
            f"ImmichAlbumCollection created: {self.path} (name: '{self.album.get('albumName', 'N/A')}', flatten: {self.flatten}, members: {len(self._members or {})})"
        )

    def _initialize_members(self):
        """Prepare the internal dictionary of members based on structure."""
        self._members = {}
        assets_list = self.album.get("assets", [])
        if not isinstance(assets_list, list):
            _logger.warning(
                f"Invalid 'assets' data in album {self.album.get('id')} for {self.path}"
            )
            return

        if self.flatten:

            for asset in assets_list:
                if not isinstance(asset, dict):
                    continue
                filename = asset.get("originalFileName")
                if filename:
                    ext = get_file_extension(filename)
                    if ext not in self.excluded_types:
                        if filename in self._members:
                            _logger.warning(
                                f"Duplicate filename '{filename}' in album {self.album.get('id')} (flat). Overwriting."
                            )
                        self._members[filename] = asset
        else:

            has_images = False
            has_videos = False
            for asset in assets_list:
                if not isinstance(asset, dict):
                    continue
                filename = asset.get("originalFileName")
                asset_type = asset.get("type")
                if filename and asset_type:
                    ext = get_file_extension(filename)
                    if ext not in self.excluded_types:
                        if asset_type == "IMAGE":
                            has_images = True
                        elif asset_type == "VIDEO":
                            has_videos = True
            if has_images:
                self._members["images"] = "IMAGE"
            if has_videos:
                self._members["videos"] = "VIDEO"

        self._member_names = sorted(self._members.keys())

    def get_member_names(self) -> List[str]:
        """Return list of member names (assets or subcollections)."""
        _logger.debug(
            f"get_member_names for {self.path} returning: {self._member_names}"
        )
        return self._member_names or []

    def get_member(self, name: str) -> Union[ImmichAsset, ImmichAssetCollection]:
        """Get a member (asset file or asset type collection)."""
        _logger.debug(
            f"get_member searching for '{name}' in {self.path}, flatten={self.flatten}"
        )

        member_info = self._members.get(name) if self._members else None

        if member_info is None:
            _logger.debug(f"Member '{name}' not found in {self.path}.")
            raise DAVError(HTTP_NOT_FOUND)

        member_path = util.join_uri(self.path, name)

        if self.flatten:

            if isinstance(member_info, dict):
                _logger.debug(f"Returning ImmichAsset for '{name}' in flat {self.path}")
                return ImmichAsset(member_path, self.environ, member_info)
            else:

                _logger.error(
                    f"Internal error: Expected dict for flat member '{name}' in {self.path}, got {type(member_info)}. Raising 500."
                )
                raise DAVError(
                    HTTP_INTERNAL_ERROR, "Internal server error: Invalid member data"
                )
        else:

            if isinstance(member_info, str) and member_info in ("IMAGE", "VIDEO"):
                _logger.debug(
                    f"Returning ImmichAssetCollection for '{name}' (type {member_info}) in nested {self.path}"
                )
                return ImmichAssetCollection(
                    member_path, self.environ, self.album, member_info
                )
            else:

                _logger.error(
                    f"Internal error: Expected 'IMAGE' or 'VIDEO' for nested member '{name}' in {self.path}, got {member_info!r}. Raising 500."
                )
                raise DAVError(
                    HTTP_INTERNAL_ERROR, "Internal server error: Invalid member data"
                )

    def get_creation_date(self) -> Optional[int]:
        return safe_isoparse_timestamp(self.album.get("createdAt"))

    def get_last_modified(self) -> Optional[int]:
        return (
            safe_isoparse_timestamp(self.album.get("updatedAt"))
            or self.get_creation_date()
        )

    def get_display_name(self) -> str:
        return self.name


class RootCollection(DAVCollection):
    """Represents the root '/' of the WebDAV share, listing albums."""

    def __init__(self, path: str, environ: dict):
        if not isinstance(path, str) or path != "/":
            _logger.error(f"RootCollection initialized with invalid path: {path!r}")
            raise ValueError("Invalid path for RootCollection, must be '/'")
        super().__init__(path, environ)
        _logger.debug("RootCollection created")

    def get_member_names(self) -> List[str]:
        """Return list of album names."""
        provider: ImmichProvider = self.provider
        names = []

        with provider.data_lock:
            if not isinstance(provider.all_album_data, list):
                _logger.error(
                    "Provider album data not available or invalid in RootCollection.get_member_names"
                )
                return []
            for entry in provider.all_album_data:
                if isinstance(entry, dict) and "albumName" in entry:
                    album_name = entry["albumName"]
                    if album_name in names:
                        _logger.warning(
                            f"Duplicate album name '{album_name}' found. Check Immich."
                        )
                    else:
                        names.append(album_name)

        _logger.debug(f"RootCollection.get_member_names returning: {sorted(names)}")
        return sorted(names)

    def get_member(self, name: str) -> ImmichAlbumCollection:
        """Get an album collection by name."""
        _logger.debug(f"RootCollection.get_member searching for album: '{name}'")
        provider: ImmichProvider = self.provider
        album_data = None

        with provider.data_lock:
            if not isinstance(provider.all_album_data, list):
                _logger.error(
                    "Provider album data not available or invalid in RootCollection.get_member"
                )
                raise DAVError(HTTP_NOT_FOUND)

            album_data = next(
                (
                    entry
                    for entry in provider.all_album_data
                    if isinstance(entry, dict) and entry.get("albumName") == name
                ),
                None,
            )

        if album_data:
            _logger.debug(f"Found album '{name}', returning ImmichAlbumCollection.")
            member_path = util.join_uri(self.path, name)
            return ImmichAlbumCollection(member_path, self.environ, album_data)
        else:
            _logger.debug(f"Album '{name}' not found in root.")
            raise DAVError(HTTP_NOT_FOUND)

    def get_creation_date(self) -> Optional[int]:
        return None

    def get_last_modified(self) -> Optional[int]:
        provider: ImmichProvider = self.provider

        with provider.data_lock:
            return (
                int(provider.last_refresh_time) if provider.last_refresh_time else None
            )

    def get_display_name(self) -> str:
        return ""


class ImmichProvider(DAVProvider):
    """WsgiDAV provider connecting to Immich API."""

    def __init__(self, config: Dict):
        super().__init__()

        self.immich_url: str = config["immich_url"].rstrip("/")
        self.api_key: str = config["api_key"]
        self.album_ids: List[str] = config["album_ids"]
        self.refresh_rate_seconds: int = config["refresh_rate_hours"] * 3600
        self.filetype_ignore_list: Set[str] = config["excluded_file_types"]
        self.flatten_structure: bool = config["flatten_structure"]

        self.data_lock = threading.Lock()
        self.all_album_data: List[Dict] = []
        self.last_refresh_time: Optional[float] = None

        self.stop_event = threading.Event()
        self.refresh_thread = threading.Thread(
            target=self._background_refresh_loop, daemon=True
        )

        _logger.info("ImmichProvider initializing...")

        api_key_display = (
            f"{self.api_key[:4]}...{self.api_key[-4:]}"
            if len(self.api_key) > 8
            else "******"
        )
        _logger.info(f"  Immich URL: {self.immich_url}")
        _logger.info(f"  API Key: {api_key_display}")
        _logger.info(
            f"  Album IDs: {'ALL' if not self.album_ids else ', '.join(self.album_ids)}"
        )
        _logger.info(f"  Refresh (hours): {config['refresh_rate_hours']}")
        _logger.info(
            f"  Exclude Types: {', '.join(self.filetype_ignore_list) if self.filetype_ignore_list else 'None'}"
        )
        _logger.info(f"  Flatten Structure: {self.flatten_structure}")

        _logger.info("Performing initial data fetch...")
        self._fetch_and_update_data()
        _logger.info("Initial data fetch complete.")

        self.refresh_thread.start()
        _logger.info("Background refresh thread started.")

    def _fetch_with_retries(
        self, url: str, max_retries: int = 3, initial_delay: float = 1.0
    ) -> Optional[Union[Dict, List]]:
        """Fetches JSON data, handling retries and errors."""
        headers = {"x-api-key": self.api_key, "Accept": "application/json"}
        delay = initial_delay
        for attempt in range(max_retries):
            response = None
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                if response.status_code == 204:
                    _logger.debug(f"Received 204 No Content from {url}")
                    return None

                if "application/json" in response.headers.get("Content-Type", ""):
                    return response.json()
                else:
                    _logger.error(
                        f"Non-JSON response from {url}: {response.status_code} {response.headers.get('Content-Type')}"
                    )
                    _logger.error(
                        f"Response text (first 500 chars): {response.text[:500]}"
                    )
                    return None
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                _logger.error(
                    f"HTTP Error fetching {url} (attempt {attempt + 1}/{max_retries}): {status_code} {e.response.reason}"
                )
                if status_code == 401:
                    _logger.critical(
                        "Received 401 Unauthorized - Please check your IMMICH_API_KEY."
                    )
                    return None
                if status_code == 404:
                    _logger.error(
                        f"Received 404 Not Found for {url}. Check URL path or ID."
                    )
                    return None

            except requests.exceptions.RequestException as e:
                _logger.error(
                    f"Request Error fetching {url} (attempt {attempt + 1}/{max_retries}): {e}"
                )
            except json.JSONDecodeError as e:
                _logger.error(
                    f"JSON Decode Error from {url} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if response:
                    _logger.error(
                        f"Response text causing decode error: {response.text[:500]}"
                    )

            if attempt < max_retries - 1:
                wait_time = delay * (2**attempt)
                _logger.info(f"Retrying fetch for {url} in {wait_time:.1f} seconds...")

                if self.stop_event.wait(wait_time):
                    _logger.info("Stop requested during retry wait.")
                    return None

        _logger.critical(f"Failed to fetch {url} after {max_retries} attempts.")
        return None

    def _get_all_album_ids_from_api(self) -> List[str]:
        """Fetch all album IDs from the Immich API."""
        url = f"{self.immich_url}/api/albums"
        _logger.debug(f"Fetching all album IDs from {url}")
        albums_list = self._fetch_with_retries(url)

        if isinstance(albums_list, list):
            ids = []
            seen_names = set()
            for album in albums_list:
                if isinstance(album, dict) and "id" in album and "albumName" in album:
                    album_id = str(album["id"])
                    album_name = album["albumName"]
                    if album_name in seen_names:
                        _logger.critical(
                            f"Duplicate album name '{album_name}' found in Immich (IDs: ... {album_id}). WebDAV access unpredictable."
                        )
                        continue
                    seen_names.add(album_name)
                    ids.append(album_id)
                else:
                    _logger.warning(f"Skipping invalid album entry in list: {album}")
            _logger.info(f"Found {len(ids)} unique album names from API.")
            return ids
        else:
            _logger.error(
                f"Failed to fetch/parse list of all albums from API (URL: {url}). Type: {type(albums_list)}"
            )
            return []

    def _fetch_and_update_data(self):
        """Fetches data for configured albums and updates the internal cache thread-safely."""
        _logger.info("Starting data refresh cycle...")
        start_time = time.time()

        target_album_ids = self.album_ids
        if not target_album_ids:
            _logger.info("Fetching all available album IDs...")
            target_album_ids = self._get_all_album_ids_from_api()
            if not target_album_ids:
                _logger.error("Failed to get any album IDs from API. Refresh aborted.")
                with self.data_lock:
                    self.all_album_data = []
                    self.last_refresh_time = time.time()
                return

        _logger.info(f"Will refresh data for {len(target_album_ids)} Album IDs.")
        new_album_data_map = {}
        processed_count = 0
        success_count = 0
        seen_album_names = set()

        for album_id in target_album_ids:
            processed_count += 1
            album_id_str = str(album_id).strip()
            if not album_id_str or album_id_str in new_album_data_map:
                continue

            url = f"{self.immich_url}/api/albums/{album_id_str}"
            _logger.debug(f"Fetching details for album {album_id_str}...")
            album_data = self._fetch_with_retries(url)

            if (
                isinstance(album_data, dict)
                and "id" in album_data
                and "albumName" in album_data
                and "assets" in album_data
                and isinstance(album_data["assets"], list)
            ):

                album_name = album_data["albumName"]
                _logger.debug(
                    f"Successfully fetched album '{album_name}' ({album_id_str}), assets: {len(album_data['assets'])}"
                )

                if album_name in seen_album_names:
                    _logger.critical(
                        f"Duplicate album name '{album_name}' found during refresh (IDs: ... vs {album_id_str}). Skipping duplicate. Rename in Immich."
                    )
                    continue

                seen_album_names.add(album_name)
                new_album_data_map[album_id_str] = album_data
                success_count += 1
            else:
                _logger.error(
                    f"Failed to fetch or validate data for album ID {album_id_str}. Skipping."
                )
                if isinstance(album_data, dict):
                    _logger.error(f"  Partial data keys: {list(album_data.keys())}")
                elif album_data is not None:
                    _logger.error(f"  Unexpected data type: {type(album_data)}")

        with self.data_lock:
            self.all_album_data = list(new_album_data_map.values())
            self.last_refresh_time = time.time()
            duration = self.last_refresh_time - start_time
            _logger.info(f"Data refresh cycle complete in {duration:.2f} seconds.")
            _logger.info(
                f"Processed {processed_count} IDs, loaded {success_count} valid albums."
            )
            total_assets = sum(len(a.get("assets", [])) for a in self.all_album_data)
            _logger.info(f"Total assets across loaded albums: {total_assets}")

    def _background_refresh_loop(self):
        """Background thread target for periodic data refresh."""
        _logger.debug("Background refresh thread started, initial wait...")

        if self.stop_event.wait(15):
            _logger.info("Background refresh thread stopped during initial wait.")
            return

        while not self.stop_event.is_set():
            _logger.info("Background refresh thread starting data refresh cycle.")
            try:
                self._fetch_and_update_data()
            except Exception as e:
                _logger.exception(f"Unhandled exception in refresh thread loop: {e}")

            wait_seconds = self.refresh_rate_seconds
            _logger.info(
                f"Background refresh thread sleeping for {wait_seconds} seconds..."
            )

            if self.stop_event.wait(wait_seconds):
                _logger.info("Stop requested during sleep interval.")
                break

        _logger.info("Background refresh thread finished.")

    def stop_refresh(self):
        """Signals the background refresh thread to stop."""
        if self.refresh_thread.is_alive():
            _logger.info("Stopping background refresh thread...")
            self.stop_event.set()
            self.refresh_thread.join(timeout=15)
            if self.refresh_thread.is_alive():
                _logger.warning("Refresh thread did not stop cleanly after 15 seconds.")
            else:
                _logger.info("Background refresh thread stopped successfully.")
        else:
            _logger.info("Background refresh thread already stopped.")

    def get_resource_inst(
        self, path: str, environ: dict
    ) -> Union[DAVCollection, DAVNonCollection]:
        """Return DAVResource object for path. Main entry point for resolution."""
        _logger.debug(f"get_resource_inst called for path: '{path}'")

        root = RootCollection("/", environ)

        try:

            resource = root.resolve(environ.get("SCRIPT_NAME", ""), path)

            if resource is None:

                _logger.error(
                    f"Resolution for '{path}' unexpectedly returned None. Raising 404."
                )
                raise DAVError(HTTP_NOT_FOUND)

            _logger.debug(
                f"get_resource_inst resolved '{path}' to: {type(resource).__name__} ({resource.path})"
            )
            return resource

        except DAVError as e:

            if e.value == HTTP_NOT_FOUND:
                _logger.debug(f"Path '{path}' not found (DAVError 404).")
            else:
                _logger.warning(
                    f"DAVError resolving '{path}': {e.value} {e.get_user_message()}"
                )
            raise

        except Exception as e:

            _logger.exception(f"Unexpected error resolving path '{path}': {e}")
            raise DAVError(
                HTTP_INTERNAL_ERROR, f"Internal server error resolving path '{path}'"
            )


def run_webdav_server():
    """Load config, initialize provider, setup WsgiDAV app, and run server."""

    load_dotenv()
    _logger.info("Loading configuration from environment variables...")

    try:
        immich_url = os.environ["IMMICH_URL"]
        api_key = os.environ["IMMICH_API_KEY"]
    except KeyError as e:
        _logger.critical(f"Missing required environment variable: {e}. Exiting.")
        return

    album_ids_env = os.getenv("ALBUM_IDS", "")
    album_ids = [id.strip() for id in album_ids_env.split(",") if id.strip()]

    try:
        refresh_rate_hours = int(os.getenv("REFRESH_RATE_HOURS", DEFAULT_REFRESH_HOURS))
    except ValueError:
        refresh_rate_hours = DEFAULT_REFRESH_HOURS

    try:
        port = int(os.getenv("WEBDAV_PORT", DEFAULT_WEBDAV_PORT))
    except ValueError:
        port = DEFAULT_WEBDAV_PORT

    excluded_types_env = os.getenv("EXCLUDED_FILE_TYPES", "")
    excluded_file_types = {
        id.strip().lower() for id in excluded_types_env.split(",") if id.strip()
    }

    flatten_structure_env = os.getenv(
        "FLATTEN_ASSET_STRUCTURE", str(DEFAULT_FLATTEN)
    ).lower()
    flatten_structure = flatten_structure_env == "true"

    provider_config = {
        "immich_url": immich_url,
        "api_key": api_key,
        "album_ids": album_ids,
        "refresh_rate_hours": refresh_rate_hours,
        "excluded_file_types": excluded_file_types,
        "flatten_structure": flatten_structure,
    }

    wsgidav_config = {
        "provider_mapping": {"/": ImmichProvider(provider_config)},
        "http_authenticator": {
            "domain_controller": None,
            "accept_basic": False,
            "accept_digest": False,
            "default_to_digest": False,
        },
        "simple_dc": {"user_mapping": {"*": True}},
        "verbose": 3,
        "props_manager": True,
        "locks_manager": True,
        "host": "0.0.0.0",
        "port": port,
        "directory_browser": {"enable": True},
        "cors": {
            "allow_origin": "*",
            "allow_methods": "GET, HEAD, POST, PUT, DELETE, OPTIONS, PROPFIND, PROPPATCH, MKCOL, COPY, MOVE, LOCK, UNLOCK",
            "allow_headers": "Content-Type, Depth, User-Agent, X-File-Size, X-Requested-With, If-Modified-Since, X-File-Name, Cache-Control, Pragma, Origin, Connection, Referer, Cookie, Authorization",
            "expose_headers": "DAV, Content-Length, Date",
            "allow_credentials": "true",
            "max_age": "1728000",
        },
    }

    _logger.info("Initializing WsgiDAVApp...")
    app = WsgiDAVApp(wsgidav_config)

    server_args = {
        "bind_addr": (wsgidav_config["host"], wsgidav_config["port"]),
        "wsgi_app": app,
    }

    _logger.info(
        f"Starting Cheroot WSGI server on http://{wsgidav_config['host']}:{wsgidav_config['port']}..."
    )
    server = wsgi.Server(**server_args)

    provider_instance = app.provider_map.get("/")

    try:
        server.start()
    except KeyboardInterrupt:
        _logger.info("Received Ctrl+C (KeyboardInterrupt)... stopping.")
    except Exception as e:
        _logger.exception(
            f"Server failed to start or encountered a critical error: {e}"
        )
    finally:
        _logger.info("Shutting down server...")
        if provider_instance and hasattr(provider_instance, "stop_refresh"):
            provider_instance.stop_refresh()
        if server and hasattr(server, "stop"):
            server.stop()
        _logger.info("Server stopped.")


if __name__ == "__main__":
    run_webdav_server()