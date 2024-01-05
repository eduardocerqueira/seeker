#date: 2024-01-05T17:03:22Z
#url: https://api.github.com/gists/ee64c1656d99e55dea9fc7be13dfd4b8
#owner: https://api.github.com/users/terjehaukaas

# ------------------------------------------------------------------------
# The following Python code is implemented by Professor Terje Haukaas at
# the University of British Columbia in Vancouver, Canada. It is made
# freely available online at terje.civil.ubc.ca together with notes,
# examples, and additional Python code. Please be cautious when using
# this code; it may contain bugs and comes without warranty of any kind.
# ------------------------------------------------------------------------

import numpy as np

def nonlinearDynamicSDOFAnalysis(duration, dt, material, M, dampingRatio, groundAcceleration, gmScaling, DDMparameters):

    numTimePoints = int(duration / dt) + 1
    t = np.linspace(0, dt * (numTimePoints-1), numTimePoints)

    # Initial declarations
    u = np.zeros(3)
    displacementOld = 0.0
    velocityOld = 0.0
    accelerationOld = 0.0

    # Scaling
    groundAcceleration *= gmScaling

    # Damping
    Kinitial = material.initialStiffness()
    C = 2 * np.sqrt(Kinitial * M) * dampingRatio

    # Newmark parameters (0.167<beta<0.25) (gamma=0.5)
    newmarkBeta = 0.25
    newmarkGamma = 0.5

    # Newmark constants
    a1 = 1.0 / (newmarkBeta * dt ** 2)
    a2 = -a1
    a3 = -1.0 / (newmarkBeta * dt)
    a4 = 1.0 - 1.0 / (2.0 * newmarkBeta)
    a5 = newmarkGamma / (newmarkBeta * dt)
    a6 = -a5
    a7 = (1.0 - newmarkGamma / newmarkBeta)
    a8 = dt * (1.0 - newmarkGamma / (2.0 * newmarkBeta))

    # Prepare to track displacements: [which-u, u-values]
    dnl = np.zeros(numTimePoints)
    uTrack = np.zeros(numTimePoints)
    vTrack = np.zeros(numTimePoints)
    aTrack = np.zeros(numTimePoints)

    # Prepare to track sensitivities: [which-u, which-x, dudx-values]
    numDDMparameters = len(DDMparameters)
    ddmDisplacementOld = np.zeros(numDDMparameters)
    ddmVelocityOld = np.zeros(numDDMparameters)
    ddmAccelerationOld = np.zeros(numDDMparameters)
    dudx = np.zeros((numDDMparameters, numTimePoints))
    dvdx = np.zeros((numDDMparameters, numTimePoints))
    dadx = np.zeros((numDDMparameters, numTimePoints))

    # INCREMENTS
    maxiter = 10
    tol = 1e-6
    for n in range(numTimePoints):

        # ITERATIONS
        converged = False
        for ii in range(maxiter):

            # Call material (notice: this is un-modified Newton-Raphson)
            F_tilde, K = material.state(u)

            # Collect terms of the residual
            F = -M * groundAcceleration[n]
            R = -F
            R += F_tilde
            massFactor = a1*u[0] + a2*displacementOld + a3*velocityOld + a4*accelerationOld
            R += M * massFactor
            dampingFactor = a5*u[0] + a6*displacementOld + a7*velocityOld + a8*accelerationOld
            R += C * dampingFactor

            # Evaluate the effective stiffness
            Keffective = K + a1 * M + a5 * C

            # Check convergence
            if np.abs(R) < tol:
                converged = True
                break

            # Solve for displacement increment
            u[2] = -R / Keffective

            # Accept convergence if the displacement increment is below machine precision
            if np.abs(u[2]) < 1e-16:
                converged = True
                break

            # Update displacements
            u[0] = u[0] + u[2]
            u[1] = u[1] + u[2]

        # Check that equilibrium iteration loop converged. If not, break load
        # step loop after reverting to previous load step that did converge.
        if not converged:
            print('\n'"No convergence with residual", R, ">", tol, "in", maxiter, "iterations for time step", n)
            import sys
            sys.exit()

        # Here we have converged, so determine velocity and acceleration at time n+1
        velocityNew = a5 * u[0] + a6 * displacementOld + a7 * velocityOld + a8 * accelerationOld
        accelerationNew = a1 * u[0] + a2 * displacementOld + a3 * velocityOld + a4 * accelerationOld

        # Store the responses (notice: total acceleration for Sa)
        dnl[n] = K/Kinitial
        uTrack[n] = u[0]
        vTrack[n] = velocityNew
        aTrack[n] = accelerationNew + groundAcceleration[n]


        # DDM sensitivity calculations, prior to commit
        if len(DDMparameters) > 0:

            # Make sure the stiffness is up-to-date
            F_tilde, K = material.state(u)
            Keffective = K + a1 * M + a5 * C

            # Loop over DDM parameters
            for ddmIndex in range(len(DDMparameters)):

                # Reset for each parameter
                ddmRHS = 0.0
                dM = 0.0
                dC = 0.0

                # Gather right-hand side contributions
                if DDMparameters[ddmIndex][0] == 'Material':
                    dF_tilde, dK, dKmdum = material.stateDerivative(u, DDMparameters[ddmIndex][1], ddmIndex, True)
                    ddmRHS -= dF_tilde
                    if DDMparameters[ddmIndex][1] == 'E':
                        dC = M / np.sqrt(M * Kinitial) * dampingRatio

                else:

                    if DDMparameters[ddmIndex][0] == 'Mass':
                        ddmRHS -= groundAcceleration[n]
                        dM = 1
                        dC = Kinitial / np.sqrt(Kinitial * M) * dampingRatio

                    elif DDMparameters[ddmIndex][0] == 'GroundMotion' and DDMparameters[ddmIndex][1] == 'Scaling':
                        ddmRHS -= M * groundAcceleration[n] / gmScaling

                    elif DDMparameters[ddmIndex][0] == 'GroundMotion' and DDMparameters[ddmIndex][1] == 'Point':
                        if n == DDMparameters[ddmIndex][2]:
                            ddmRHS -= gmScaling * M

                    elif DDMparameters[ddmIndex][0] == 'Damping':
                        dC = 2 * np.sqrt(Kinitial * M)

                    else:
                        print('\n'"Error: Cannot handle DDM option")
                        import sys
                        sys.exit()

                    # Must call the material even if the parameter isn't in it
                    dF_tilde, dK, dKmdum = material.stateDerivative(u, 'Void', ddmIndex, False)
                    ddmRHS -= dF_tilde

                # (a2*M + a6*C)*dun/dtheta + (a3*M + a7*C)*dudotn/dtheta + (a4*M + a8*C)*dudotdotn/dtheta
                ddmRHS -= M * (a2 * ddmDisplacementOld[ddmIndex] + a3 * ddmVelocityOld[ddmIndex] + a4 * ddmAccelerationOld[ddmIndex])
                ddmRHS -= C * (a6 * ddmDisplacementOld[ddmIndex] + a7 * ddmVelocityOld[ddmIndex] + a8 * ddmAccelerationOld[ddmIndex])

                # dM/dtheta * (a1*un1 + a2*un + a3*udotn + a4*udotdotn)
                ddmRHS -= dM * (a1 * u[0] + a2 * displacementOld + a3 * velocityOld + a4 * accelerationOld)

                # Store the parenthesis that multiplies with the derivative of the damping matrix
                ddmRHS -= dC * (a5 * u[0] + a6 * displacementOld + a7 * velocityOld + a8 * accelerationOld)

                # Solve for the derivative of the displacements
                ddmDisplacementNew = ddmRHS / Keffective

                # Update the derivatives of velocity and acceleration responses
                ddmAccelerationNew = a1 * ddmDisplacementNew + a2 * ddmDisplacementOld[ddmIndex] + a3 * ddmVelocityOld[ddmIndex] + a4 * ddmAccelerationOld[ddmIndex]
                ddmVelocityNew = a5 * ddmDisplacementNew + a6 * ddmDisplacementOld[ddmIndex] + a7 * ddmVelocityOld[ddmIndex] + a8 * ddmAccelerationOld[ddmIndex]

                # Now that we're done; reset the sensitivity vectors at the previous step
                ddmDisplacementOld[ddmIndex] = ddmDisplacementNew
                ddmVelocityOld[ddmIndex] = ddmVelocityNew
                ddmAccelerationOld[ddmIndex] = ddmAccelerationNew

                # Pick up the derivative that will be plotted
                ddmu = ddmDisplacementNew
                dudx[ddmIndex, n] = ddmu
                dvdx[ddmIndex, n] = ddmVelocityNew
                dadx[ddmIndex, n] = ddmAccelerationNew

                # Store unconditional derivatives
                if DDMparameters[ddmIndex][0] == 'Material':
                    material.commitSensitivity(u, ddmu, DDMparameters[ddmIndex][1], ddmIndex, True)
                else:
                    material.commitSensitivity(u, ddmu, 'Void', ddmIndex, False)

        # Commit and update material state using converged displacement
        material.commit()

        # Set response values at time step i equal to response at time n+1
        displacementOld = u[0]
        velocityOld = velocityNew
        accelerationOld = accelerationNew

        # Zero second column of displacement vector after commit
        u[1] = 0.0

    return t, uTrack, vTrack, aTrack, dudx, dvdx, dadx, dnl