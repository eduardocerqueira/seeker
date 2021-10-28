#date: 2021-10-28T17:08:17Z
#url: https://api.github.com/gists/2609fe63add7542f42cf68a25a134dda
#owner: https://api.github.com/users/craabreu

import numpy as np

from simtk import openmm, unit


class BussiDonadioParrinelloIntegrator(openmm.CustomIntegrator):
    def __init__(self, temperature, frictionCoeff, stepSize, dof):
        self._dof_is_odd = dof % 2 == 1
        self._shape = (dof - 2 + dof % 2)/2
        self._rng = np.random.default_rng(None)
        super().__init__(stepSize)
        self._add_global_variables(temperature, frictionCoeff)
        self.addUpdateContextState()
        self._add_boost('dt')
        self._add_translation('0.5*dt')
        self._add_rescaling('dt')
        self._add_translation('0.5*dt')

    def _add_global_variables(self, temperature, frictionCoeff):
        self.addPerDofVariable('x1', 0)
        self.addGlobalVariable('sumRsq', 0)
        self.addGlobalVariable('mvv', 0)
        self.addGlobalVariable('kT', unit.MOLAR_GAS_CONSTANT_R*temperature)
        self.addGlobalVariable('friction', frictionCoeff)

    def _add_translation(self, timestep):
        self.addComputePerDof('x', f'x+{timestep}*v')
        self.addComputePerDof('x1', 'x')
        self.addConstrainPositions()
        self.addComputePerDof('v', f'v+(x-x1)/({timestep})')
        self.addConstrainVelocities()

    def _add_boost(self, timestep):
        self.addComputePerDof('v', f'v+{timestep}*f/m')
        self.addConstrainVelocities()

    def _add_rescaling(self, timestep):
        rescaling = 'v*sqrt(A+C*B*(R1^2+sumRsq)+2*sqrt(C*B*A)*R1)'
        rescaling += '; R1=gaussian'
        rescaling += '; C=kT/mvv'
        rescaling += '; B=1-A'
        rescaling += f'; A=exp(-friction*{timestep})'
        self.addComputeSum('mvv', 'm*v*v')
        self.addComputePerDof('v', rescaling)

    def setRandomNumberSeed(self, seed):
        self._rng = np.random.default_rng(seed)
        super().setRandomNumberSeed(seed)

    def step(self, steps):
        sumRsq = 2*self._rng.standard_gamma(self._shape, size=steps)
        if self._dof_is_odd:
            sumRsq += self._rng.standard_normal(size=steps)
        for value in sumRsq:
            self.setGlobalVariableByName('sumRsq', value)
            super().step(1)
