#date: 2024-05-22T17:05:08Z
#url: https://api.github.com/gists/80995362e2ebbff818437f4d3e60efe3
#owner: https://api.github.com/users/canismarko

In [1]: from haven.instrument.xray_source import PlanarUndulator

In [2]: from ophyd import sim

In [3]: undulator = sim.instantiate_fake_device(PlanarUndulator)
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:325, in Component.__get__(self, instance, owner)
    324 try:
--> 325     return instance._signals[self.attr]
    326 except KeyError:

KeyError: 'energy'

During handling of the above exception, another exception occurred:

KeyError                                  Traceback (most recent call last)
File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:325, in Component.__get__(self, instance, owner)
    324 try:
--> 325     return instance._signals[self.attr]
    326 except KeyError:

KeyError: 'actuate'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:1352, in Device._instantiate_component(self, attr)
   1351 try:
-> 1352     self._signals[attr] = cpt.create_component(self)
   1353     sig = self._signals[attr]

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:266, in Component.create_component(self, instance)
    265     pv_name = self.maybe_add_prefix(instance, "suffix", self.suffix)
--> 266     cpt_inst = self.cls(pv_name, parent=instance, **kwargs)
    267 else:

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/signal.py:619, in DerivedSignal.__init__(self, derived_from, write_access, name, parent, **kwargs)
    618 if isinstance(derived_from, str):
--> 619     derived_from = getattr(parent, derived_from)
    621 # Metadata keys from the class itself take precedence

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:1327, in Device.__getattr__(self, name)
   1326 if "." in name:
-> 1327     return operator.attrgetter(name)(self)
   1329 # Components will be instantiated through the descriptor mechanism in
   1330 # the Component class, so anything reaching this point is an error.

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:1331, in Device.__getattr__(self, name)
   1329 # Components will be instantiated through the descriptor mechanism in
   1330 # the Component class, so anything reaching this point is an error.
-> 1331 raise AttributeError(name)

AttributeError: _prefixEnergyparent

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
Cell In[3], line 1
----> 1 undulator = sim.instantiate_fake_device(PlanarUndulator)

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/sim.py:1307, in instantiate_fake_device(dev_cls, name, prefix, **specified_kw)
   1305 kwargs["name"] = name if name is not None else dev_cls.__name__
   1306 kwargs["prefix"] = prefix
-> 1307 return dev_cls(**kwargs)

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:894, in Device.__init__(self, prefix, name, kind, read_attrs, configuration_attrs, parent, **kwargs)
    890     self.configuration_attrs = list(configuration_attrs)
    892 with do_not_wait_for_lazy_connection(self):
    893     # Instantiate non-lazy signals and lazy signals with subscriptions
--> 894     [
    895         getattr(self, attr)
    896         for attr, cpt in self._sig_attrs.items()
    897         if not cpt.lazy or cpt._subscriptions
    898     ]

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:895, in <listcomp>(.0)
    890     self.configuration_attrs = list(configuration_attrs)
    892 with do_not_wait_for_lazy_connection(self):
    893     # Instantiate non-lazy signals and lazy signals with subscriptions
    894     [
--> 895         getattr(self, attr)
    896         for attr, cpt in self._sig_attrs.items()
    897         if not cpt.lazy or cpt._subscriptions
    898     ]

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:327, in Component.__get__(self, instance, owner)
    325     return instance._signals[self.attr]
    326 except KeyError:
--> 327     return instance._instantiate_component(self.attr)

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:1352, in Device._instantiate_component(self, attr)
   1342     raise RuntimeError(
   1343         f"The Component {attr!r} exists at the Python level and "
   1344         "has triggered the `_instantiate_component` "
   (...)
   1348         "a Component but does not inherent from Device."
   1349     ) from None
   1351 try:
-> 1352     self._signals[attr] = cpt.create_component(self)
   1353     sig = self._signals[attr]
   1354     for event_type, functions in cpt._subscriptions.items():

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:266, in Component.create_component(self, instance)
    264 if self.suffix is not None:
    265     pv_name = self.maybe_add_prefix(instance, "suffix", self.suffix)
--> 266     cpt_inst = self.cls(pv_name, parent=instance, **kwargs)
    267 else:
    268     cpt_inst = self.cls(parent=instance, **kwargs)

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/pv_positioner.py:90, in PVPositioner.__init__(self, prefix, limits, name, read_attrs, configuration_attrs, parent, egu, **kwargs)
     78 def __init__(
     79     self,
     80     prefix="",
   (...)
     88     **kwargs,
     89 ):
---> 90     super().__init__(
     91         prefix=prefix,
     92         read_attrs=read_attrs,
     93         configuration_attrs=configuration_attrs,
     94         name=name,
     95         parent=parent,
     96         **kwargs,
     97     )
     99     if self.__class__ is PVPositioner:
    100         raise TypeError(
    101             "PVPositioner must be subclassed with the correct "
    102             "signals set in the class definition."
    103         )

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:894, in Device.__init__(self, prefix, name, kind, read_attrs, configuration_attrs, parent, **kwargs)
    890     self.configuration_attrs = list(configuration_attrs)
    892 with do_not_wait_for_lazy_connection(self):
    893     # Instantiate non-lazy signals and lazy signals with subscriptions
--> 894     [
    895         getattr(self, attr)
    896         for attr, cpt in self._sig_attrs.items()
    897         if not cpt.lazy or cpt._subscriptions
    898     ]

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:895, in <listcomp>(.0)
    890     self.configuration_attrs = list(configuration_attrs)
    892 with do_not_wait_for_lazy_connection(self):
    893     # Instantiate non-lazy signals and lazy signals with subscriptions
    894     [
--> 895         getattr(self, attr)
    896         for attr, cpt in self._sig_attrs.items()
    897         if not cpt.lazy or cpt._subscriptions
    898     ]

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:327, in Component.__get__(self, instance, owner)
    325     return instance._signals[self.attr]
    326 except KeyError:
--> 327     return instance._instantiate_component(self.attr)

File /home/beams1/B268176/miniforge3/envs/haven/lib/python3.9/site-packages/ophyd/device.py:1361, in Device._instantiate_component(self, attr)
   1357             sig.subscribe(method, event_type=event_type, run=sig.connected)
   1358 except AttributeError as ex:
   1359     # Raise a different Exception, as AttributeError will be shadowed
   1360     # during initial access
-> 1361     raise RuntimeError(
   1362         f"AttributeError while instantiating " f"component: {attr}"
   1363     ) from ex
   1365 return sig

RuntimeError: AttributeError while instantiating component: actuate