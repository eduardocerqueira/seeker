#date: 2023-12-06T17:08:03Z
#url: https://api.github.com/gists/9c8c0a1bd5d5015b287d9ffb3f259b49
#owner: https://api.github.com/users/jacopoabramo

# adapted from https://github.com/bluesky/tutorials/blob/main/archive/04%20-%20Array%20Detector.ipynb

import os
import threading
import itertools

from ximea.xiapi import Camera as XCamera, Image as XImage
from ophyd import Device, Component, Signal, DeviceStatus
from ophyd.areadetector.filestore_mixins import resource_factory
from bluesky import RunEngine
from databroker.v2 import temp
from bluesky.plans import count

import numpy
from pathlib import Path
from PIL import Image

def store_image(filepath: str, image: numpy.ndarray) -> tuple:
    """
    This function should integrate directly with the hardware.
    
    No concepts particular to ophyd are involved here.
    Just tell the hardware to take an image, however that works.
    This function should block until acquisition is complete or
    raise if acquisition fails.
    
    It will be run on a worker thread, so it will not block
    ophyd / the RunEngine.
    """
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # Save the image.
    Image.fromarray(image).save(filepath)
    return image.shape

class ExternalFileReference(Signal):
    """
    A pure software signal pointing to data in an external file
    
    The parent device is intended to set the value of this Signal to a datum_id.
    """
    def __init__(self, *args, shape: tuple, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape

    def describe(self):
        res = super().describe()
        # Tell consumers that readings from this Signal point to "external" data,
        # data that is not in-line in the reading itself.
        res[self.name].update(dict(external="FILESTORE:", dtype="array", shape=self.shape))
        return res

class Camera(Device):
    """
    An ophyd device for a camera that acquires images and saves them in files.
    """
    # We initialize the shape to [] and update it below once we know the shape
    # of the array.
    image = Component(ExternalFileReference, value="", kind="normal", shape=[])

    def __init__(self, *args, root_path, **kwargs):
        super().__init__(*args, **kwargs)
        self._root_path = root_path
        # Use this lock to ensure that we only process one "trigger" at a time.
        # Generally bluesky should care of this, so this is just an extra
        # precaution.
        self._acquiring_lock = threading.Lock()
        self._counter = None  # set to an itertools.count object when staged
        # Accumulate Resource and Datum documents in this cache.
        self._asset_docs_cache = []
        # This string is included in the Resource documents to indicate which
        # can of reader ("handler") is needed to access the relevant data.
        self._SPEC = "MY_FORMAT_SPEC"
        
        self._handle = XCamera()
        self._buffer = XImage()
        
        # TODO: find a way to modify Camera settings
        self._handle.open_device()
        self._handle.set_exposure(10000) # 10 ms exposure time

    def stage(self):
        # Set the filepath where will be saving images.
        self._rel_path_template = f"images/ximea_img_%d.jpg"
        # Create a Resource document referring to this series of images that we
        # are about to take, and stash it in _asset_docs_cache.
        resource, self._datum_factory = resource_factory(
            self._SPEC, self._root_path, self._rel_path_template, {}, "posix")
        self._asset_docs_cache.append(("resource", resource))
        self._counter = itertools.count()
        self._handle.start_acquisition()
        return super().stage()

    def unstage(self):
        self._handle.stop_acquisition()
        self._counter = None
        self._asset_docs_cache.clear()
        return super().unstage()

    def trigger(self):
        status = DeviceStatus(self)
        if self._counter is None:
            raise RuntimeError("Device must be staged before triggering.")
        i = next(self._counter)
        # Start a background thread to capture an image and write it to disk.
        thread = threading.Thread(target=self._capture, args=(status, i))
        thread.start()
        # Promptly return a status object, which will be marked "done" when the
        # capture completes.
        return status

    def _capture(self, status, i):
        "This runs on a background thread."
        try:
            if not self._acquiring_lock.acquire(timeout=0):
                raise RuntimeError("Cannot trigger, currently triggering!")
            filepath = os.path.join(self._root_path, self._rel_path_template % i)
            # Kick off requests, or subprocess, or whatever with the result
            # that a file is saved at `filepath`.
            self._handle.get_image(self._buffer)
            shape = store_image(filepath, self._buffer.get_image_data_numpy())
            self.image.shape = shape
            # Compose a Datum document referring to this specific image, and
            # stash it in _asset_docs_cache.
            datum = self._datum_factory({"index": i})
            self._asset_docs_cache.append(("datum", datum))
            self.image.set(datum["datum_id"]).wait()
            
        except Exception as exc:
            status.set_exception(exc)
        else:
            status.set_finished()
        finally:
            self._acquiring_lock.release()

    def collect_asset_docs(self):
        "Yield the documents from our cache, and reset it."
        yield from self._asset_docs_cache
        self._asset_docs_cache.clear()

class MyHandler:
    def __init__(self, resource_path):
        # resource_path is really a template string with a %d in it
        self._template = resource_path

    def __call__(self, index):
        import PIL, numpy
        filepath = str(self._template) % index
        return numpy.asarray(PIL.Image.open(filepath))

print("Staging camera...", end="")
camera = Camera(root_path="external_data", name="camera")
print(" done")

print("Staging run engine and catalog...", end="")
RE = RunEngine()
db = temp()
RE.subscribe(db.v1.insert)
print(" done")

db.register_handler("MY_FORMAT_SPEC", MyHandler)

print("Staging run engine execution...", end="")
RE(count(detectors = [camera], num = 10))
print(" execution ended")

run = db[-1]  # Acccess the most recent run.
dataset = run.primary.read()  # Access the dataset of its 'primary' stream.
print(dataset)