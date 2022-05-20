#date: 2022-05-20T17:10:07Z
#url: https://api.github.com/gists/43638dec44f58768f0c2356ab74efe8d
#owner: https://api.github.com/users/Horolsky

import numpy as np
from numpy import linalg
from typing import Any, List
from fractions import Fraction as Frac
import csv
import os.path

DECIMALS = 3

MRCI = lambda n: (.0, .0, .0, .52, .89, 1.11, 1.25, 1.35, 1.40, 1.45, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59)[n]
"Mean Random Consistency Index"

CT = lambda n: (.0, .0, .0, .05, .08, .1)[n] if n <= 5 else .1
"Consistency Threshold"

GCIT = lambda n: (.0, .0, .0, .1573, .3526, .37)[n] if n <= 5 else .37
"GCI Threshold"

class iNormalisation:
  "PCM Normalisation interface"

  @staticmethod
  def MRCI(mt: np.ndarray):
    "Mean Random Consistency Index"
    return MRCI(mt.shape[0])

  @classmethod
  def CT(cls, mt: np.ndarray):
    "Consistency Threshold"
    return CT(mt.shape[0])

  @classmethod
  def V(cls, mt: np.ndarray):
    "vector of unnormalised column weignts"
    raise NotImplementedError('method of abstract class')

  @classmethod
  def W(cls, mt: np.ndarray):
    "vector of normalised column weights"
    V = cls.V(mt)
    return np.round(V / V.sum(), DECIMALS)

  @classmethod
  def CI(cls, mt: np.ndarray):
    "Consistency Index"
    raise NotImplementedError('method of abstract class')

  @classmethod
  def CR(cls, mt: np.ndarray):
    "Consistency Ratio"
    return np.round(cls.CI(mt) / cls.MRCI(mt), DECIMALS)

class EM(iNormalisation):
  "Eigenvector normalisation methods"
  @classmethod
  def V(cls, mt: np.ndarray):
    "vector of unnormalised AN weignts"
    eigenvalues, eigenvectors = linalg.eig(mt)
    vec = eigenvectors[:, eigenvalues.argmax()]
    if vec.sum() < 0:
      vec *= -1
    return np.real_if_close(np.round(vec, DECIMALS))

  @classmethod
  def CI(cls, mt: np.ndarray):
    "Eigenvector Consistency Index"
    eigenvalues, _ = linalg.eig(mt)
    l = eigenvalues.max()
    n = mt.shape[0]
    return np.real_if_close(np.round((l - n) / (n - 1), DECIMALS))

class RGMM(iNormalisation):
  "Row geometric mean method"

  @classmethod
  def CT(cls, mt: np.ndarray):
    "Consistency Threshold"
    return GCIT(mt.shape[0])

  @classmethod
  def V(cls, mt: np.ndarray):
    "vector of unnormalised AN weignts"
    n = mt.shape[0]
    return np.round(np.power(mt.prod(axis=1), 1/n), DECIMALS)

  @classmethod
  def CI(cls, mt: np.ndarray):
    "Geometric Consistency Index"
    n = mt.shape[0]
    w = cls.W(mt)
    e = [ mt[i,j] * w[j] / w[i] for i in range(n) for j in range(i+1, n)]
    sm = np.power(np.log(e),2).sum()
    return np.round(2 * sm / ((n - 1) * (n - 2)), DECIMALS)

  @classmethod
  def CR(cls, mt: np.ndarray):
    "Geometric Consistency Ratio not defined for RGMM, returns GCI for inheritance consistency"
    return cls.CI(mt)

class AN(iNormalisation):
  "Additive normalisation methods"
  @classmethod
  def V(cls, mt: np.ndarray):
    "vector of unnormalised AN weignts"
    # reciprocals of column sums
    return np.round(1 / mt.sum(axis=0), DECIMALS)

  @classmethod
  def CI(cls, mt: np.ndarray):
    "Harmonic Consistency Index"
    n = mt.shape[0]
    HM = mt.shape[0] / cls.V(mt).sum() # harmonic means of columns
    return np.round(((HM - n) * (n + 1)) / (n * (n - 1)), DECIMALS)

class PCM:
  "Pairwise comparison matrix class"

  __methods = {
      'EM': EM,
      'AN': AN,
      'RGMM': RGMM
      }

  _data: np.ndarray
  _name: str
  _method_name: str
  _method: iNormalisation

  @staticmethod
  def read_csv(path, method: str = "AN"):
    with  open(path, newline='') as csvfile:
      dialect = csv.Sniffer().sniff(csvfile.read(1024))
      csvfile.seek(0)
      reader = csv.reader(csvfile, dialect)
      rational = lambda s: float(Frac(s) or 0)
      name = os.path.split(path)[1]
      name = name[:name.rfind('.')]
      return PCM([[rational(item) for item in row] for row in reader], name, method)

  def __init__(self, data: Any, name: str, method: str = "AN"):
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[0] == data.shape[1]
    assert method in PCM.__methods

    self._data = data
    self._name = name
    self._method_name = method
    self._method = PCM.__methods.get(method)

  @property
  def n(self):
    return self._data.shape[0]

  @property
  def data(self):
    return self._data

  @property
  def name(self):
    return self._name

  @property
  def method(self):
    "normalisation method"
    return self._method_name

  def reset_method(self, method: str):
    assert method in PCM.__methods
    self._method_name = method
    self._method = PCM.__methods.get(method)

  @property
  def V(self):
    "vector of unnormalised column weignts"
    return self._method.V(self._data)

  @property
  def W(self):
    "vector of normalised column weignts"
    return self._method.W(self._data)

  @property
  def CI(self):
    "Consistency Index"
    return self._method.CI(self._data)

  @property
  def CR(self):
    "Consistency Ratio"
    return self._method.CR(self._data)

  @property
  def consistent(self) -> bool:
    "PCM is practically consistent"
    return self.CR <= self._method.CT(self._data)

  @property
  def report(self):
    return f"""
      pcm: {self.name}
      method: {self.method}
      n: {self.n}
      V: {self.V}
      W: {self.W}
      CI: {self.CI}
      CR: {self.CR}
      is practically consistent: {'yes' if self.consistent else 'no'}
    """

class Synthesis:
  __methods = {'distributive', 'ideal', 'multiplicative'}

  @classmethod
  def dist(cls, Va, wc):
    "global weights of alternatives mt"
    r = Va / Va.sum(axis=0) # normalize by column axis
    return (wc * r).sum(axis=1)

  @classmethod
  def ideal(cls, Va, wc):
    "global weights of alternatives mt"
    raise NotImplementedError

  @classmethod
  def mult(cls, Va, wc):
    "global weights of alternatives mt"
    raise NotImplementedError

  def __init__(self, method = 'distributive'):
    assert method in Synthesis.__methods
    self._function = {
        'distributive': Synthesis.dist,
        'ideal': Synthesis.ideal,
        'multiplicative': Synthesis.mult
        }.get(method)

  def __call__(self, Va, wc):
    return self._function(Va, wc)

class HierarchyBranch:
  """
  single branch of Analytic Hierarchy Process
  includes one root element (criteria)
  and its children elements (alternatives)
  """
  def __init__(self, criteria_pcm: PCM, alternatives_pcm: List[PCM], method = 'distributive'):

    synthesis = Synthesis(method)
    assert len(alternatives_pcm) == criteria_pcm.n
    assert isinstance(criteria_pcm, PCM)

    nof_alts = alternatives_pcm[0].n
    for alt in alternatives_pcm:
      assert isinstance(alt, PCM)
      assert alt.n == nof_alts

    self._C = criteria_pcm
    self._A = alternatives_pcm
    self._nof_alts = nof_alts
    self._method = method
    self._synth = synthesis

  @property
  def method(self):
    return self._method

  @property
  def nof_alts(self):
    return self._nof_alts

  @property
  def C(self):
    "criteria PCM"
    return self._C

  def A(self, criteria_index: int):
    "alternative PCM"
    return self._A[criteria_index]

  @property
  def Wglob(self):
    "alternatives global weights"
    Va = np.array([A.V for A in self._A]).T
    wc = self._C.W
    return self._synth(Va, wc)

  @property
  def alt_rank(self):
    "alternatives ranking by global weights"
    W = self.Wglob
    order = W.argsort()
    rng = [ 0 for _ in range(len(order))]
    for i, j in enumerate(order):
      rng[j] = i
    return np.array(rng)
