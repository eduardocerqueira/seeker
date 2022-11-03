#date: 2022-11-03T17:06:57Z
#url: https://api.github.com/gists/6a275a9dbdeda57a4f7545145ef714ba
#owner: https://api.github.com/users/kratsg

import numpy as np
import onnx
import onnxruntime
import uproot
import h5py
import json

model="./SimpleAnalysisCodes/data/ThreeBjets_NN_2020_model.onnx"

session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open("./SimpleAnalysisCodes/data/ThreeBjets_NN_2020_config.json") as fp:
    config = json.load(fp)

branches = config['branches']
parameters = np.array([tuple(item.values()) for item in config['parameters']])

nparameters = parameters.shape[0]

mean = np.array(config['normalization']['mean'])
std = np.array(config['normalization']['stddev'])

with uproot.open("./ThreeBjets_NN_2020.root") as fp:
    for chunk in fp['ntuple'].iterate(expressions=branches[:-3], library="np", how=tuple):
        step_size = len(chunk[0])
        for parameter in parameters:
            isGtt, mGluino, mLSP = parameter
            intermediate = np.tile(parameter, [step_size, 1])
            data = np.concatenate([np.stack(chunk).T, intermediate], axis=1)
            data -= mean
            data /= std
            result = session.run([output_name], {input_name: data.astype('float32')})[0]

            # make a mask for only events we record information for
            mask = chunk[0] == 0
            result_py = result[~mask]

            NNbranch = f'NNoutput_{isGtt:d}_{mGluino:d}_{mLSP:d}'
            result_cpp = np.array(fp['ntuple'][NNbranch].array()[~mask], dtype='float32')

            match = np.allclose(result_py, result_cpp)
            print(f'For {NNbranch}, do the results match?: {match}')
            if not match:
                isclose = np.isclose(result_py, result_cpp)
                print(f'  - {np.argwhere(~isclose)}')
                indices = np.where(~isclose)
                print(f'  - py:  {result_py[indices]}')
                print(f'  - cpp: {result_cpp[indices]}')
