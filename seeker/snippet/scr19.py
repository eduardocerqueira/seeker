#date: 2022-02-15T17:01:10Z
#url: https://api.github.com/gists/551ec718a2ddb97370a8609307dae1c1
#owner: https://api.github.com/users/vaibhavtmnit

textdataloader = DataLoader(textdataset,batch_size = 2)

for i in textdataloader:
    print(i)
    
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-61-dde51e430dfa> in <module>
----> 1 for i in textdataloader:
      2     print(i)

/opt/anaconda3/lib/python3.8/site-packages/torch/utils/data/dataloader.py in __next__(self)
    433         if self._sampler_iter is None:
    434             self._reset()
--> 435         data = self._next_data()
    436         self._num_yielded += 1
    437         if self._dataset_kind == _DatasetKind.Iterable and \

/opt/anaconda3/lib/python3.8/site-packages/torch/utils/data/dataloader.py in _next_data(self)
    473     def _next_data(self):
    474         index = self._next_index()  # may raise StopIteration
--> 475         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    476         if self._pin_memory:
    477             data = _utils.pin_memory.pin_memory(data)

/opt/anaconda3/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     45         else:
     46             data = self.dataset[possibly_batched_index]
---> 47         return self.collate_fn(data)

/opt/anaconda3/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
     81             raise RuntimeError('each element in list of batch should be of equal size')
     82         transposed = zip(*batch)
---> 83         return [default_collate(samples) for samples in transposed]
     84 
     85     raise TypeError(default_collate_err_msg_format.format(elem_type))

/opt/anaconda3/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py in <listcomp>(.0)
     81             raise RuntimeError('each element in list of batch should be of equal size')
     82         transposed = zip(*batch)
---> 83         return [default_collate(samples) for samples in transposed]
     84 
     85     raise TypeError(default_collate_err_msg_format.format(elem_type))

/opt/anaconda3/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
     53             storage = elem.storage()._new_shared(numel)
     54             out = elem.new(storage)
---> 55         return torch.stack(batch, 0, out=out)
     56     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
     57             and elem_type.__name__ != 'string_':

RuntimeError: stack expects each tensor to be equal size, but got [3] at entry 0 and [4] at entry 1