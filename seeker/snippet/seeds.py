#date: 2024-12-04T17:00:01Z
#url: https://api.github.com/gists/1a077b9daa4d4c98ac75877466433861
#owner: https://api.github.com/users/riga

# coding: utf-8

"""
Pseudo code for generating deterministic event and jet seeds,
to be used in random number generators with consistent output.

The seeds use the entire uint64 space and are proven to be
uniformly distributed using bijective hashing.

It requires a list of the first 200 primes plus some event
and object level integer (!) quantities. Floats are not used
as any algorithm based on them is intrinsically non-deterministic.
"""

# first 200 primes, starting at 2
primes = [...]

# singular hash function converting int to uint64
def create_seed(val: int, n_hex: int = 16) -> int:
  return int(hashlib.sha256(bytes(str(val), "utf-8")).hexdigest()[:-(n_hex + 1):-1], base=16)

#
# event seed calculation
#

# inputs to identifiy the event (order matters!)
# (get() is a placeholder for a retrieval function)
index_inputs = [
    get("event"), get("run"), get("luminosityBlock"),
]

# event-level inputs, i.e, one number per event (order matters!)
event_inputs = [
    get("Pileup.nPU"),
    get("nJet"), get("nFatJet"), get("nSubJet"),
    get("nPhoton"), get("nMuon"), get("nElectron"), get("nTau"),
    get("nSV"), get("nGenJet"),
]

# object-level inputs, i.e., one number per object (order matters!)
# (here, each get() would return a list of numbers)
object_inputs = [
    get("Electron.jetIdx"), get("Electron.seediPhiOriY"),
    get("Muon.jetIdx"), get("Muon.nStations"),
    get("Tau.jetIdx"), get("Tau.decayMode"),
    get("Jet.nConstituents"), get("Jet.nElectrons"), get("Jet.nMuons"),
]

# start by creating a short seed from index inputs
event_seed = create_seed(
    index_input[0] * primes[7] + index_input[1] * primes[5] + index_input[2] * primes[3],
)

# fold with event level info
value_offset = 3
prime_offset = 15
for i, inp in enumerate(event_inputs):
    inp += i + value_offset
    event_seed += primes[(inp + prime_offset) % len(primes)] * inp

# fold with object level info
for i, inps in enumerate(object_inputs):
    inp_sum = 0
    for j, inp in enumerate(inps):
        inp += i + value_offset
        inp_sum += inp + inp * (j + 1) + inp**2 * (j + 1)
    event_seed += primes[(inp_sum + prime_offset) % len(primes)] * inp_sum

# final seed
event_seed = create_seed(event_seed)  # done

#
# jet seed calculation
#

for i, jet in enumerate(jets):
    jet_seed = event_seed + primes[event_seed % len(primes)] * (i + primes[50])
    jet_seed = create_seed(jet_seed)  # done
