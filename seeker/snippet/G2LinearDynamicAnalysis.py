#date: 2024-01-05T16:58:12Z
#url: https://api.github.com/gists/da077a8a5360894ce543219063c05efb
#owner: https://api.github.com/users/terjehaukaas

# ------------------------------------------------------------------------
# The following Python code is implemented by Professor Terje Haukaas at
# the University of British Columbia in Vancouver, Canada. It is made
# freely available online at terje.civil.ubc.ca together with notes,
# examples, and additional Python code. Please be cautious when using
# this code; it may contain bugs and comes without warranty of any kind.
# ------------------------------------------------------------------------
# Notation:
# ndof     = number of free DOFs
# ntot     = total number of all  DOFs
# free     = range of free DOFs (1 to ndof)
# nelem    = number of elements in elemlist
# elemlist = list of elements
# ua       = vector of displacements for all DOFs
# ug       = vector of displacements for global element DOFs
# Fa       = nodal load vector in model for all DOFs
# Fa_tilde = restoring force vector for all DOFs
# Ka       = stiffness matrix for all DOFs
# Kf       = stiffness matrix for free DOFs
# R        = residual force vector (Fa_tilde-Fa) for free DOFs

from scipy.linalg import eig, lu_factor, lu_solve
import numpy as np
import matplotlib.pyplot as plt

def linearDynamicAnalysis(model, dampingModel, groundMotion, dt, duration, trackNode, trackDOF, DDMparameters):

    # Plot the model
    model.plotModel()

    # Get data from the model
    ndof, ntot, F, Ma, elemlist = model.getData()
    free = range(ndof)
    nelem = len(elemlist)

    # Initialize global vectors and matrix
    Fa_tilde = np.zeros(ntot)
    ua = np.zeros((ntot,3))
    Ka = np.zeros((ntot, ntot))

    # Initialize element state
    for i in range(nelem):
        id, xyz, ug = model.localize(i, ua)
        element = elemlist[i]
        element.initialize(xyz)

    # Loop over elements to assemble stiffness matrix and element load vector
    for i in range(nelem):
        id, xyz, ug = model.localize(i, ua)
        element = elemlist[i]
        Fg_tilde, Kg = element.state(xyz, ug, 1.0)
        Ka[np.ix_(id, id)] = Ka[np.ix_(id,id)] + Kg
        Fa_tilde[id] = Fa_tilde[id] + Fg_tilde

    # Mass matrix and stiffness matrix for free DOFs
    Mf = Ma[np.ix_(free, free)]
    Kf = Ka[np.ix_(free, free)]
    det = np.linalg.det(Kf)
    if det < 1.0e-10:
        print('\n'"ERROR: The determinant of the stiffness matrix is", det)
        import sys
        sys.exit()

    # ------------------------------------------------------------------------
    # EIGENVALUE ANALYSIS
    # ------------------------------------------------------------------------

    # Solve eigenvalue problem
    allEigenInformation = eig(Kf, Mf, left=True, right=False)

    # Sort the eigenvalues
    allEigenvalues = allEigenInformation[0]
    sortedEigenvalueIndices = np.argsort(np.abs(allEigenvalues))
    eigenvalues = []
    eigenvectors = []
    for i in range(len(sortedEigenvalueIndices)):
        if allEigenvalues[sortedEigenvalueIndices[i]].real == np.inf:
            pass
        else:
            eigenvalues.append(allEigenvalues[sortedEigenvalueIndices[i]].real)
            eigenvectors.append(allEigenInformation[1][:,sortedEigenvalueIndices[i]])

    # Normalize eigenvectors
    for i in range(len(eigenvalues)):
        norm = np.linalg.norm(eigenvectors[i])
        eigenvectors[i] = eigenvectors[i] / norm

    # Calculate frequencies and periods
    numEigenvalues = len(eigenvalues)
    naturalFrequencies = []
    naturalPeriods = []
    print('\n'"Found the following", numEigenvalues, "eigenvalues:"'\n')
    for i in range(numEigenvalues):
        frequency = np.sqrt(eigenvalues[i])
        naturalFrequencies.append(frequency)
        period = 2 * np.pi / frequency
        naturalPeriods.append(period)
        print("   Frequency: %.2frad, Period: %.3fsec" % (frequency, period))

    # ------------------------------------------------------------------------
    # DAMPING MATRIX
    # ------------------------------------------------------------------------

    if dampingModel[1] != 'Initial':
        print('\n'"Error: Something else than initial stiffness specified for damping in linear dynamic analysis.")
        import sys
        sys.exit()

    if dampingModel[0] == 'Rayleigh':

        if dampingModel[2] == 'GivenCs':

            cM = dampingModel[3]
            cK = dampingModel[4]
            Cf = np.multiply(Mf, cM) + np.multiply(Kf, cK)

        elif dampingModel[2] == 'UseEigenvalues':

            # Enforce that damping at user-selected frequencies
            omega1 = naturalFrequencies[dampingModel[3] - 1]
            omega2 = naturalFrequencies[dampingModel[4] - 1]

            # Calculate cM and cK in C = cM*M + cK*K
            factor = 2 * dampingModel[5] / (omega1 + omega2)
            cM = omega1 * omega2 * factor
            cK = factor

            # Damping matrix
            Cf = np.multiply(Mf, cM) + np.multiply(Kf, cK)

            # Calculate the damping ratio at the different periods
            dampingRatios = []
            for i in range(numEigenvalues):
                dampingRatios.append(cM / (2 * naturalFrequencies[i]) + cK * naturalFrequencies[i] / 2)

            # Plot those damping ratios
            plt.ion()
            plt.figure()
            plt.title("Damping Ratio at Different Natural Periods")
            plt.plot(naturalPeriods, dampingRatios, 'ks-', label='Actual damping')
            plt.plot([naturalPeriods[0], naturalPeriods[-1]], [dampingModel[5], dampingModel[5]], 'r-', label='Target damping')
            plt.legend()
            plt.xlabel("Period [sec]")
            plt.ylabel("Damping Ratio")
            print('\n'"Click somewhere in the plot to continue...")
            plt.waitforbuttonpress()

        else:
            print('\n'"Error: Cannot understand the damping model")
            import sys
            sys.exit()

    elif dampingModel[0] == 'Modal':

        Sum = np.zeros((len(free), len(free)))
        for i in range(numEigenvalues):
            modalMass = (eigenvectors[i].dot(Mf)).dot(eigenvectors[i])
            outerProduct = np.outer(eigenvectors[i], eigenvectors[i])
            Sum += 2.0 * dampingModel[2] * np.sqrt(eigenvalues[i]) / modalMass * outerProduct

        Cf = (Mf.dot(Sum)).dot(Mf)

    else:
        print('\n'"Error: Cannot understand the damping model")
        import sys
        sys.exit()

    # ------------------------------------------------------------------------
    # READ GROUND MOTION [Unit of g]
    # ------------------------------------------------------------------------

    # Load the ground motion record
    groundAcceleration = []
    f = open(groundMotion, "r")
    lines = f.readlines()
    for oneline in lines:
        splitline = oneline.split()
        for j in range(len(splitline)):
            value = 9.81 * float(splitline[j])
            groundAcceleration.append(value)

    # Output
    print('\n'"Max ground acceleration in units of g:", np.max(np.abs(groundAcceleration)) / 9.81)

    # Create time axis
    numTimePoints = int(duration / dt)
    if numTimePoints > len(groundAcceleration):
        for i in range(numTimePoints - len(groundAcceleration)):
            groundAcceleration.append(0.0)
    t = np.linspace(0, dt * numTimePoints, numTimePoints)

    # Plot the ground motion
    plt.ion()
    plt.figure()
    plt.title("Ground Motion Record")
    plt.plot(t, groundAcceleration[0:len(t)], 'k-')
    plt.xlabel("Time [sec]")
    plt.ylabel("Ground Acceleration [m/s^2]")
    print('\n'"Click somewhere in the plot to continue...")
    plt.waitforbuttonpress()

    # Mass participation vector times mass matrix
    iotaMf = -np.diag(Mf)

    # ------------------------------------------------------------------------
    # MDOF NEWMARK ALGORITHM
    # ------------------------------------------------------------------------

    # Initial declarations
    displacementOld = np.zeros(ndof)
    velocityOld = np.zeros(ndof)
    accelerationOld = np.zeros(ndof)
    ddmDisplacementOld = np.zeros((ndof, len(DDMparameters)))
    ddmVelocityOld = np.zeros((ndof, len(DDMparameters)))
    ddmAccelerationOld = np.zeros((ndof, len(DDMparameters)))
    ddmua = np.zeros(ntot)

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

    # Establish effective stiffness and "those factors"
    newDispFactor = np.multiply(a1, Mf) + np.multiply(a5, Cf)
    Keffective = newDispFactor + Kf
    dispFactor = np.multiply(a2, Mf) + np.multiply(a6, Cf)
    velocFactor = np.multiply(a3, Mf) + np.multiply(a7, Cf)
    accelFactor = np.multiply(a4, Mf) + np.multiply(a8, Cf)

    # Do the LU factorization
    LU = lu_factor(Keffective)

    # Identify the index of the displacement to be tracked
    index = model.DOF[trackNode - 1, trackDOF - 1]

    # Start the loop
    trackDisp = []
    uB = 0
    uC = 0
    maxPeak = 0
    minPeak = 0
    maxPeakTime = 0
    minPeakTime = 0
    trackAllDDMs = np.zeros((numTimePoints, len(DDMparameters)))
    for step in range(numTimePoints):

        # Set the external force vector
        Fnew = np.multiply(iotaMf, groundAcceleration[step])

        # Establish the right-hand side of the system of equations
        rhsNew = Fnew - dispFactor.dot(displacementOld) - velocFactor.dot(velocityOld) - accelFactor.dot(accelerationOld)

        # Solve the system of equilibrium equations
        displacementNew = lu_solve(LU, rhsNew)

        # Determine acceleration at time step i+1
        accelerationNew = np.multiply(a1, displacementNew) + np.multiply(a2, displacementOld) + np.multiply(a3, velocityOld) + np.multiply(a4, accelerationOld)

        # Determine velocity at time step i+1
        velocityNew = np.multiply(a5, displacementNew) + np.multiply(a6, displacementOld) + np.multiply(a7, velocityOld) + np.multiply(a8, accelerationOld)

        # Store the displacement at the tracked node
        disp = displacementNew[index]
        trackDisp.append(disp)

        # DDM sensitivity analysis
        if len(DDMparameters) > 0:

            # Make sure we have the correct displacements
            ua[free, 0] = displacementNew

            for ddmIndex in range(len(DDMparameters)):

                # Initialize for each parameter
                theDampingRatioIsTheDDMparameter = False
                dKa = np.zeros((ntot, ntot))
                dMa = np.zeros((ntot, ntot))
                ddmRHSa = np.zeros(ntot)
                dcMdtheta = 0.0
                dcKdtheta = 0.0

                # Gather right-hand side contributions from elements
                if DDMparameters[ddmIndex][0] == 'Element':
                    for eleNum in DDMparameters[ddmIndex][2]:
                        i = eleNum - 1
                        id, xyz, ug = model.localize(i, ua)
                        element = elemlist[i]
                        ddmRHSg, dKg, dKgdug = element.stateDerivative(xyz, ug, 1.0, DDMparameters[ddmIndex][1], 0, True)
                        ddmRHSa[id] = ddmRHSa[id] + ddmRHSg
                        dKa[np.ix_(id, id)] = dKa[np.ix_(id,id)] + dKg

                elif DDMparameters[ddmIndex][0] == 'Node' and DDMparameters[ddmIndex][1] == 'M':
                    dMa = model.getMassDerivative(DDMparameters[ddmIndex][2])
                    ddmRHSa = np.multiply(np.diag(dMa), groundAcceleration[step])

                elif DDMparameters[ddmIndex][0] == 'Model' and DDMparameters[ddmIndex][1] == 'targetDamping':
                    theDampingRatioIsTheDDMparameter = True

                elif DDMparameters[ddmIndex][0] == 'Model' and DDMparameters[ddmIndex][1] == 'cM':
                    dcMdtheta = 1.0
                    if dampingModel[2] == 'UseEigenvalues':
                        print('\n'"Error: cM or cK is a DDM parameter, but it's calculated, not given.")
                        import sys
                        sys.exit()

                elif DDMparameters[ddmIndex][0] == 'Model' and DDMparameters[ddmIndex][1] == 'cK':
                    dcKdtheta = 1.0
                    if dampingModel[2] == 'UseEigenvalues':
                        print('\n'"Error: cM or cK is a DDM parameter, but it's calculated, not given.")
                        import sys
                        sys.exit()
                else:
                    print('\n'"Error: Cannot handle DDM parameter of type", DDMparameters[ddmIndex][0])
                    import sys
                    sys.exit()

                # Derivative of mass and stiffness matrix
                dKf = dKa[np.ix_(free, free)]
                dMf = dMa[np.ix_(free, free)]

                # dF/dtheta - dKdtheta*un1
                ddmRHSf = -ddmRHSa[free]

                # (a2*M + a6*C)*dun/dtheta + (a3*M + a7*C)*dudotn/dtheta + (a4*M + a8*C)*dudotdotn/dtheta
                ddmRHSf -= dispFactor.dot(ddmDisplacementOld[:,ddmIndex]) + velocFactor.dot(ddmVelocityOld[:,ddmIndex]) + accelFactor.dot(ddmAccelerationOld[:,ddmIndex])

                # dM/dtheta * (a1*un1 + a2*un + a3*udotn + a4*udotdotn)
                ddmRHSf -= dMf.dot(a1 * displacementNew + a2 * displacementOld + a3 * velocityOld + a4 * accelerationOld)

                # Store the parenthesis that multiplies with the derivative of the damping matrix
                a5parenthesis = a5 * displacementNew + a6 * displacementOld + a7 * velocityOld + a8 * accelerationOld

                # Damping derivative for dC/dtheta * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                if dampingModel[0] == 'Rayleigh':

                    # cM * dM/dtheta * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                    ddmRHSf -= cM * dMf.dot(a5parenthesis)

                    # cK * dK/dtheta * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                    ddmRHSf -= cK * dKf.dot(a5parenthesis)

                    # dcM/dtheta*M*(a5*un1+a6*un+a7*udotn+a8*udotdotn) if cM is the DDM parameter
                    ddmRHSf -= dcMdtheta * Mf.dot(a5parenthesis)

                    # dcM/dtheta*M*(a5*un1+a6*un+a7*udotn+a8*udotdotn) if cK is the DDM parameter
                    ddmRHSf -= dcKdtheta * Kf.dot(a5parenthesis)

                    if dampingModel[2] == 'UseEigenvalues':

                        # Check whether the target damping ratio is the DDM parameter
                        if theDampingRatioIsTheDDMparameter:
                            dfactor = 2 / (omega1 + omega2)
                            dcM = omega1 * omega2 * dfactor
                            dcK = dfactor

                        else:
                            # Derivative of eigenvalues and eigenvectors
                            domegas = []
                            for i in range(numEigenvalues):
                                lhs = (eigenvectors[i].dot(Mf)).dot(eigenvectors[i])
                                rhs = (eigenvectors[i].dot(dKf)).dot(eigenvectors[i]) - \
                                      eigenvalues[i] * (eigenvectors[i].dot(dMf)).dot(eigenvectors[i])
                                dgammadtheta = rhs/lhs
                                domegas.append(0.5 / np.sqrt(eigenvalues[i]) * dgammadtheta)

                            # Derivative of those damping factors
                            domega1 = domegas[dampingModel[3] - 1]
                            domega2 = domegas[dampingModel[4] - 1]
                            dfactor = -2 * dampingModel[5] / (omega1 + omega2)**2 * (domega1 + domega2)
                            dcM = domega1 * omega2 * factor + omega1 * domega2 * factor + omega1 * omega2 * dfactor
                            dcK = dfactor

                        # dcM/dtheta * M * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                        ddmRHSf -= dcM * Mf.dot(a5parenthesis)

                        # dcK/dtheta * K * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                        ddmRHSf -= dcK * Kf.dot(a5parenthesis)

                elif dampingModel[0] == 'Modal':

                    if theDampingRatioIsTheDDMparameter:
                        Sum = np.zeros((len(free), len(free)))
                        for i in range(numEigenvalues):
                            modalMass = (eigenvectors[i].dot(Mf)).dot(eigenvectors[i])
                            outerProduct = np.outer(eigenvectors[i], eigenvectors[i])
                            Sum += 2.0 * np.sqrt(eigenvalues[i]) / modalMass * outerProduct

                        dCf = (Mf.dot(Sum)).dot(Mf)

                    else:

                        # Derivative of eigenvalues and eigenvectors
                        dgammas = []
                        dvectors = []
                        for i in range(numEigenvalues):
                            lhs = (eigenvectors[i].dot(Mf)).dot(eigenvectors[i])
                            rhs = (eigenvectors[i].dot(dKf)).dot(eigenvectors[i]) - \
                                  eigenvalues[i] * (eigenvectors[i].dot(dMf)).dot(eigenvectors[i])
                            dgammas.append(rhs/lhs)
                            coefficientMatrix = np.subtract(Kf, np.multiply(eigenvalues[i], Mf))
                            rhsParenthesis = np.multiply(dgammas[i], Mf) + np.multiply(eigenvalues[i], dMf) - dKf
                            dvectors.append(np.linalg.pinv(coefficientMatrix).dot(rhsParenthesis).dot(eigenvectors[i]))

                        # Derivative of the damping matrix
                        term1 = np.zeros((len(free), len(free)))
                        term2 = np.zeros((len(free), len(free)))
                        term3 = np.zeros((len(free), len(free)))
                        term4 = np.zeros((len(free), len(free)))
                        term5 = np.zeros((len(free), len(free)))
                        term6 = np.zeros((len(free), len(free)))
                        for i in range(numEigenvalues):
                            modalMass = (eigenvectors[i].dot(Mf)).dot(eigenvectors[i])
                            dmodalmass = (dvectors[i].dot(Mf)).dot(eigenvectors[i]) + \
                                         (eigenvectors[i].dot(dMf)).dot(eigenvectors[i]) + \
                                         (eigenvectors[i].dot(Mf)).dot(dvectors[i])
                            outer1 = np.outer(eigenvectors[i], eigenvectors[i])
                            outer2 = np.outer(dvectors[i], eigenvectors[i])
                            outer3 = np.outer(eigenvectors[i], dvectors[i])
                            term1 = term1 + 2.0 * dampingModel[2] * np.sqrt(eigenvalues[i]) / modalMass * outer1
                            term2 = term2 + 2.0 * dampingModel[2] * 0.5 / np.sqrt(eigenvalues[i]) * dgammas[i] / modalMass * outer1
                            term3 = term3 + 2.0 * dampingModel[2] * np.sqrt(eigenvalues[i]) / modalMass**2 * dmodalmass * outer1
                            term4 = term4 + 2.0 * dampingModel[2] * np.sqrt(eigenvalues[i]) / modalMass * outer2
                            term5 = term5 + 2.0 * dampingModel[2] * np.sqrt(eigenvalues[i]) / modalMass * outer3
                            term6 = term6 + 2.0 * dampingModel[2] * np.sqrt(eigenvalues[i]) / modalMass * outer1

                        dCf = (dMf.dot(term1)).dot(Mf)
                        dCf += (Mf.dot(term2)).dot(Mf)
                        dCf -= (Mf.dot(term3)).dot(Mf)
                        dCf += (Mf.dot(term4)).dot(Mf)
                        dCf += (Mf.dot(term5)).dot(Mf)
                        dCf += (Mf.dot(term6)).dot(dMf)

                    # dC/dtheta * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                    ddmRHSf -= dCf.dot(a5parenthesis)

                else:
                    print('\n'"Error: Cannot understand the damping model")
                    import sys
                    sys.exit()

                # Solve for the derivative of the displacements
                ddmDisplacementNew = lu_solve(LU, ddmRHSf)

                # Update the derivatives of velocity and acceleration responses
                ddmAccelerationNew = np.multiply(a1, ddmDisplacementNew) + np.multiply(a2, ddmDisplacementOld[:,ddmIndex]) + np.multiply(a3, ddmVelocityOld[:,ddmIndex]) + np.multiply(a4, ddmAccelerationOld[:,ddmIndex])
                ddmVelocityNew = np.multiply(a5, ddmDisplacementNew) + np.multiply(a6, ddmDisplacementOld[:,ddmIndex]) + np.multiply(a7, ddmVelocityOld[:,ddmIndex]) + np.multiply(a8, ddmAccelerationOld[:,ddmIndex])

                # Now that we're done; reset the sensitivity vectors at the previous step
                ddmDisplacementOld[:,ddmIndex] = ddmDisplacementNew
                ddmVelocityOld[:,ddmIndex] = ddmVelocityNew
                ddmAccelerationOld[:,ddmIndex] = ddmAccelerationNew

                # Pick up the derivative that will be plotted
                ddmua[free] = ddmDisplacementNew
                trackAllDDMs[step, ddmIndex] = ddmua[index]

        # Keep track of overall max & min response
        uA = uB
        uB = uC
        uC = disp
        if uB > uA and uB > uC and uB > maxPeak:
            maxPeak = uB
            maxPeakTime = t[step-1]
        elif uB < uA and uB < uC and uB < minPeak:
            minPeak = uB
            minPeakTime = t[step-1]

        # Set response values at time step i equal to response at time step i+1
        displacementOld = displacementNew
        velocityOld = velocityNew
        accelerationOld = accelerationNew

    # ------------------------------------------------------------------------
    # PLOT THE RESPONSE & RETURN
    # ------------------------------------------------------------------------

    if maxPeak == 0:
        maxPeak = np.max(trackDisp)
        maxPeakTime = t[len(t)-1]
    if minPeak == 0:
        minPeak = np.min(trackDisp)
        minPeakTime = t[len(t)-1]

    plt.ion()
    plt.figure()
    plt.autoscale(True)
    plt.title("Displacement (max=%.4f at %.2f, min=%.4f at %.2f)" % (maxPeak, maxPeakTime, minPeak, minPeakTime))
    plt.plot(t, trackDisp, 'k-')
    plt.xlabel("Time [sec]")
    plt.ylabel("Displacement [m]")
    print('\n'"Click somewhere in the plot to continue...")
    plt.waitforbuttonpress()

    if len(DDMparameters) > 0:
        return trackDisp, trackAllDDMs
    else:
        return trackDisp, 0.0