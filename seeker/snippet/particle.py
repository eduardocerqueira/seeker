#date: 2022-07-29T16:53:57Z
#url: https://api.github.com/gists/baa5223652cee6458c8913b7bf66d941
#owner: https://api.github.com/users/jonnymind

#!/bin/env python3

from cmath import pi
import random

count = 1000000

print("""Bell's experiment simulation
We'll run {} photons pairs through three detectors placed at a 60 deg angle, detecting their spin 
on two out of three detectors chosen at random (they could be the same).

Bell's disequation indicates that, if the spins are random and perfectly entangled, the two
particles should be detected as having opposite spins 5/9 of the times. 

However, there is no such thing as a perfect detector; the detection will make the photon to
align itself along the detection axis (if 'up'), or in the opposite direction (if 'down'). The
entangled photon will switch accordingly.

Given this, we expect that a practical experiment will find opposite spins half of the times,
which is what actual real-world experiments demonstrate.
""".format(count))

class Detector:
    def __init__(self, angle, tolerance):
        self.angle = angle
        self.tolerance = tolerance

    def detect(self, spin):
        if( (self.angle + spin) % 2 < 1+self.tolerance):
            return "up"
        return "down"
        

# Create three detextors at 0, 2/3 and 4/3 radians.
# Notice: a circle is two radians long.
detectors = [Detector(0, 0), Detector(2/3, 0), Detector(4/3, 0)]


same_theory = 0
same_practice = 0

for angle in range(count):
    # Pick two random detectors
    detector1 = detectors[random.randint(0, len(detectors)-1)]
    detector2 = detectors[random.randint(0, len(detectors)-1)]

    # Created two conjugated particles, with defined but unknown spins
    spin_angle_1 = random.random() * 2 
    spin_angle_2 = (spin_angle_1 + 1) % 2
    
    # Detect - the two particles, as they are in nature.  
    spin_alpha = detector1.detect(spin_angle_1)
    spin_beta = detector2.detect(spin_angle_2)

    # The first detector will actually flip the particle along its axis.
    if spin_alpha == "up":
        spin_angle_detected_1 = detector1.angle
    else:
        spin_angle_detected_1 = (detector1.angle + 1) % 2

    # And since the particles are entagled, the other will flip too
    spin_angle_detected_2 = ( spin_angle_detected_1 + 1) % 2

    # So, the spin detected by the second detector will now be...
    spin_gamma = detector2.detect(spin_angle_detected_2)

    # How many different spin we should theoretically observe if we could magically "see" the spin?
    if(spin_alpha != spin_beta):
        same_theory += 1
    
    # But since detection requires interference, how many we'll see in an experiment?
    if(spin_alpha != spin_gamma):
        same_practice += 1

print("Theoretical conjugated spin ratio: {}".format(same_theory/count))
print("Experimental conjugated spin ratio: {}".format(same_practice/count))
