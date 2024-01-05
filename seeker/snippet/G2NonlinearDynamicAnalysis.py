#date: 2024-01-05T17:01:16Z
#url: https://api.github.com/gists/1c6db73b8f8a46ab770869fd2279789f
#owner: https://api.github.com/users/terjehaukaas

# ------------------------------------------------------------------------
# The following Python code is implemented by Professor Terje Haukaas at
# the University of British Columbia in Vancouver, Canada. It is made
# freely available online at terje.civil.ubc.ca together with notes,
# examples, and additional Python code. Please be cautious when using
# this code; it may contain bugs and comes without warranty of any kind.
# ------------------------------------------------------------------------
# Notation:
# ndof = number of free DOFs
# ntot = total number of all  DOFs
# free = range of free DOFs (1 to ndof)
# ua   = matrix of displacements for "all" DOFs
#        First column of ua  = total displacements
#        Second column of ua = displacements since last committed state
#        Third column of ua  = solution to system of equations

from scipy.linalg import eig, lu_factor, lu_solve
import numpy as np
import matplotlib.pyplot as plt

def checkSymmetry(matrix):
    return np.allclose(matrix, matrix.T, rtol=1e-05, atol=1e-08)

def eigenInformation(K, M, output=False, justK=False):

    if justK:
        allEigenInformation = eig(K, left=True, right=False)
    else:
        allEigenInformation = eig(K, M, left=True, right=False)

    allEigenvalues = allEigenInformation[0]
    sortedEigenvalueIndices = np.argsort(np.abs(allEigenvalues))
    gammas = []
    omegas = []
    vectors = []
    if output:
        print('\n'"Found the following eigenvalues:"'\n')
    for i in range(len(sortedEigenvalueIndices)):
        if allEigenvalues[sortedEigenvalueIndices[i]].real == np.inf:
            pass
        else:
            gamma = allEigenvalues[sortedEigenvalueIndices[i]].real
            gammas.append(gamma)
            omega = np.sqrt(gamma)
            omegas.append(omega)
            vectors.append(allEigenInformation[1][:, sortedEigenvalueIndices[i]])
            period = 2 * np.pi / omega
            if output:
                print("   Frequency: %.2frad, Period: %.3fsec" % (omega, period))

    return gammas, omegas, vectors

def eigenDerivativesWrtTheta(K, dK, M, dM, gammas, vectors):

    dgammasdtheta = []
    domegasdtheta = []
    dvectorsdtheta = []
    for i in range(len(gammas)):

        # dgamma/dtheta
        lhs = (vectors[i].dot(M)).dot(vectors[i])
        rhs = (vectors[i].dot(dK)).dot(vectors[i]) - \
              gammas[i] * (vectors[i].dot(dM)).dot(vectors[i])
        gammaDerivative = rhs / lhs
        dgammasdtheta.append(gammaDerivative)

        # domega/dtheta
        domegasdtheta.append(0.5 / np.sqrt(gammas[i]) * gammaDerivative)

        # dvector/dtheta
        coefficientMatrix = np.subtract(K, np.multiply(gammas[i], M))
        rhsParenthesis = np.multiply(dgammasdtheta[i], M) + np.multiply(gammas[i], dM) - dK
        dvectorsdtheta.append(np.linalg.pinv(coefficientMatrix).dot(rhsParenthesis).dot(vectors[i]))

    return dgammasdtheta, domegasdtheta, dvectorsdtheta


def eigenDerivativesWrtK(K, M, gammas, vectors):

    dgammasdK = []
    domegasdK = []
    dvectorsdK = []
    for mode in range(len(gammas)):

        # dgamma/dK
        lhs = (vectors[mode].dot(M)).dot(vectors[mode])
        rhs = np.outer(vectors[mode], vectors[mode])
        gammaDerivative = rhs / lhs
        dgammasdK.append(gammaDerivative)
        domegasdK.append(0.5 / np.sqrt(gammas[mode]) * gammaDerivative)

        # Pseudo-inverse of coefficient matrix
        pseudoInverse = np.linalg.pinv(np.subtract(K, np.multiply(gammas[mode], M)))

        # dphi/dK
        gammaTimesM = np.einsum('kl,ij->klij', gammaDerivative, M)
        dKdK = np.einsum('ik,jl->ijkl', np.eye(len(K)), np.eye(len(K)))
        onedvectordK = np.einsum('pi,klij,j->pkl', pseudoInverse, (gammaTimesM - dKdK), vectors[mode])
        dvectorsdK.append(onedvectordK)

    return dgammasdK, domegasdK, dvectorsdK


def nonlinearDynamicAnalysis(model, dampingModel, groundMotion, gmScaling, dtgm, dt, duration, trackNodes, trackDOFs, DDMparameters, plotFlag=True):

    # Newton-Raphson setup
    maxiter = 100
    tol = 1e-6

    # Plot the model
    if plotFlag:
        model.plotModel()

    # Check input
    error = False
    if dt > dtgm:
        error = True
    if error:
        print('\n'"Error in input to the nonlinear dynamic analysis algorithm")
        import sys
        sys.exit()

    # Get data from the model
    ndof, ntot, Fa, Ma, elemlist = model.getData()
    free = range(ndof)
    nelem = len(elemlist)

    # Initialize vectors and matrices
    Fa_tilde = np.zeros(ntot)
    ua = np.zeros((ntot,3))
    Ka = np.zeros((ntot, ntot))

    # Initialize element states
    for i in range(nelem):
        id, xyz, ug = model.localize(i, ua)
        element = elemlist[i]
        element.initialize(xyz)

    # Initial stiffness matrix
    for i in range(nelem):
        id, xyz, ug = model.localize(i, ua)
        element = elemlist[i]
        Fg_tilde, Kg = element.state(xyz, ug, 1.0)
        Ka[np.ix_(id, id)] = Ka[np.ix_(id,id)] + Kg

    # Mass and stiffness along free DOFs
    Mf = Ma[np.ix_(free, free)]
    Kf = Ka[np.ix_(free, free)]
    KfInitial = Kf

    # Check structural integrity
    det = np.linalg.det(Kf)
    if det < 1.0e-10:
        print('\n'"ERROR: The determinant of the stiffness matrix is", det)
        import sys
        sys.exit()

    # ------------------------------------------------------------------------
    # READ GROUND MOTION [Unit of g]
    # ------------------------------------------------------------------------

    # Load the ground motion record
    rawGroundMotion = []
    f = open(groundMotion, "r")
    lines = f.readlines()
    for oneline in lines:
        splitline = oneline.split()
        for j in range(len(splitline)):
            value = 9.81 * float(splitline[j]) * gmScaling
            rawGroundMotion.append(value)

    # Output
    if plotFlag:
        print('\n'"Maximum ground acceleration in units of g: %.2f" % (np.max(np.abs(rawGroundMotion)) / 9.81))

    # Lay out time axis and determine ground acceleration at times when analysis is conducted
    numTimePoints = int(duration / dt) + 1
    t = np.linspace(0, dt * (numTimePoints-1), numTimePoints)
    groundAcceleration = []
    for i in range(numTimePoints):
        analysisTime = i*dt
        for j in range(len(rawGroundMotion)):
            gmTime = j*dtgm
            if analysisTime == gmTime:
                groundAcceleration.append(rawGroundMotion[j])
            elif analysisTime > gmTime and analysisTime < round(gmTime+dtgm, 8) and j < len(rawGroundMotion)-1:
                previousGM = rawGroundMotion[j]
                nextGM = rawGroundMotion[j+1]
                groundAcceleration.append(previousGM + (nextGM-previousGM)/dtgm * (analysisTime-gmTime))
    for i in range(len(t) - len(groundAcceleration)):
        groundAcceleration.append(0.0)

    # Plot the ground motion
    if plotFlag:
        plt.ion()
        plt.figure()
        plt.title("Ground Motion Record")
        plt.plot(t, groundAcceleration, 'k-')
        plt.xlabel("Time [sec]")
        plt.ylabel("Ground Acceleration [m/s^2]")
        print('\n'"Click somewhere in the plot to continue...")
        plt.waitforbuttonpress()

    # Mass participation vector times mass matrix
    iotaMf = -np.diag(Mf)

    # ------------------------------------------------------------------------
    # MDOF NEWMARK & NEWTON-RAPHSON
    # ------------------------------------------------------------------------

    # Initial declarations
    displacementOld = np.zeros(ndof)
    velocityOld = np.zeros(ndof)
    accelerationOld = np.zeros(ndof)

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
    if len(trackNodes) is not len(trackDOFs):
        print("Error: Inconsistent number of nodes and DOFs to track the response for.")
        import sys
        sys.exit()
    numDispsToTrack = len(trackNodes)
    uTrack = np.zeros((numDispsToTrack, numTimePoints))
    vTrack = np.zeros((numDispsToTrack, numTimePoints))
    aTrack = np.zeros((numDispsToTrack, numTimePoints))
    trackedIndices = []
    for i in range(numDispsToTrack):
        trackedIndices.append(model.DOF[int(trackNodes[i]-1), int(trackDOFs[i]-1)])

    # Prepare to track sensitivities: [which-u, which-x, dudx-values]
    numDDMparameters = len(DDMparameters)
    dudx = 0
    dvdx = 0
    dadx = 0
    if numDDMparameters > 0:
        ddmua = np.zeros(ntot)
        ddmDisplacementOld = np.zeros((ndof, numDDMparameters))
        ddmVelocityOld = np.zeros((ndof, numDDMparameters))
        ddmAccelerationOld = np.zeros((ndof, numDDMparameters))
        dudx = np.zeros((numDispsToTrack, numDDMparameters, numTimePoints))
        dvdx = np.zeros((numDispsToTrack, numDDMparameters, numTimePoints))
        dadx = np.zeros((numDispsToTrack, numDDMparameters, numTimePoints))

    # INCREMENTS
    dnl1 = []
    dnl2 = []
    determinant = []
    firstEigenvalueOfK = []
    reportDamping = True
    reportRayleigh = True
    for n in range(numTimePoints):

        # ITERATIONS
        for ii in range(maxiter):

            # Initialize state storage
            Fa_tilde[:] = 0.0
            Ka[:] = 0.0

            # Loop over all elements
            for j in range(nelem):
                id, xyz, ue = model.localize(j, ua)
                element = elemlist[j]
                Fg_tilde, Kg = element.state(xyz, ue, 1)
                Fa_tilde[id] = Fa_tilde[id] + Fg_tilde
                Ka[np.ix_(id, id)] = Ka[np.ix_(id, id)] + Kg

            # Check the new stiffness
            Kf = Ka[np.ix_(free, free)]
            det = np.linalg.det(Kf)
            if det < 1.0e-10:
                print('\n'"ERROR: The determinant of the stiffness matrix is", det)
                import sys
                sys.exit()

            # Solve the eigenvalue problem and track the degree of nonlinearity
            gammas, omegas, vectors = eigenInformation(Kf, Mf, reportDamping and plotFlag)
            reportDamping = False
            omega1ForDegreeOfNonlinearity = omegas[0]
            omega2ForDegreeOfNonlinearity = omegas[1]
            if ii==0 and n==0:
                firstOmega1 = omegas[0]
                firstOmega2 = omegas[1]

            # Damping matrix
            if dampingModel[0] == 'Rayleigh':

                if (dampingModel[1] == 'Initial' and ii==0 and n==0) or dampingModel[1] == 'Current':

                    # Keep the stiffness matrix used in the damping model for later use
                    KfDamping = Kf

                    if dampingModel[2] == 'Given':

                        cM = dampingModel[3]
                        cK = dampingModel[4]

                    elif dampingModel[2] == 'Current':

                        # Keep eigen-information too
                        gammasDamping = gammas
                        omegasDamping = omegas
                        vectorsDamping = vectors

                        # Enforce that damping at user-selected frequencies
                        omega1 = omegas[dampingModel[3]-1]
                        omega2 = omegas[dampingModel[4]-1]

                        # Calculate cM and cK in C = cM*M + cK*K
                        factor = 2 * dampingModel[5] / (omega1 + omega2)
                        cM = omega1 * omega2 * factor
                        cK = factor

                    elif dampingModel[2] == 'Initial':

                        # This is the RTi option
                        gammas, omegas, vectors = eigenInformation(KfInitial, Mf, False)
                        gammasDamping = gammas
                        omegasDamping = omegas
                        vectorsDamping = vectors
                        omega1 = omegas[dampingModel[3]-1]
                        omega2 = omegas[dampingModel[4]-1]
                        factor = 2 * dampingModel[5] / (omega1 + omega2)
                        cM = omega1 * omega2 * factor
                        cK = factor

                    else:
                        print('\n'"Error: Cannot understand the damping model")
                        import sys
                        sys.exit()

                    # Damping matrix
                    Cf = np.multiply(Mf, cM) + np.multiply(KfDamping, cK)

                    # Calculate the damping ratio at the different periods
                    dampingRatios = []
                    for k in range(len(omegas)):
                        dampingRatios.append(cM / (2.0 * omegas[k]) + cK * omegas[k] / 2.0)

                    # Plot those damping ratios
                    if reportRayleigh and plotFlag:
                        reportRayleigh = False
                        plt.ion()
                        plt.figure()
                        plt.title("Damping Ratio at Natural Periods")
                        plt.plot(np.divide(2*np.pi, omegas), dampingRatios, 'ko')
                        plt.xlabel("Period [sec.]")
                        plt.ylabel("Damping ratio, $\zeta$")
                        print('\n'"Click somewhere in the plot to continue...")
                        plt.waitforbuttonpress()

            elif dampingModel[0] == 'Modal':

                if (dampingModel[1] == 'Initial' and ii==0 and n==0) or dampingModel[1] == 'Current':

                    # Keep the information that was used to calculate the damping
                    KfDamping = Kf
                    gammasDamping = gammas
                    omegasDamping = omegas
                    vectorsDamping = vectors

                    # Calculate the sum between the mass matrices
                    Sum = np.zeros((len(free), len(free)))
                    for i in range(len(omegas)):
                        modalMass = (vectors[i].dot(Mf)).dot(vectors[i])
                        outerProduct = np.outer(vectors[i], vectors[i])
                        Sum = Sum + 2.0 * dampingModel[2] * omegas[i] / modalMass * outerProduct

                    # Pre- and post-multiply by the mass matrix
                    Cf = (Mf.dot(Sum)).dot(Mf)

            else:
                print('\n'"Error: Cannot understand the damping model")
                import sys
                sys.exit()

            # Collect terms of the residual
            Rf = Fa_tilde[free]
            Ff = np.multiply(iotaMf, groundAcceleration[n])
            Rf -= Ff
            massFactor = a1*ua[free, 0] + a2*displacementOld + a3*velocityOld + a4*accelerationOld
            Rf += Mf.dot(massFactor)
            dampingFactor = a5*ua[free, 0] + a6*displacementOld + a7*velocityOld + a8*accelerationOld
            Rf += Cf.dot(dampingFactor)

            # Check convergence
            residualNorm = np.linalg.norm(Rf)
            if residualNorm < tol:
                break

            # Evaluate the effective stiffness
            Keffective = Kf + a1 * Mf + a5 * Cf

            # Solve the system of equations
            #LU = lu_factor(Keffective)
            #ua[free, 2] = lu_solve(LU, -Rf)
            ua[free, 2] = np.linalg.solve(Keffective, -Rf)

            # Update displacements
            ua[:, 0] = ua[:, 0] + ua[:, 2]
            ua[:, 1] = ua[:, 1] + ua[:, 2]

        # Check that equilibrium iteration loop converged. If not, break load
        # step loop after reverting to previous load step that did converge.
        if residualNorm > tol:
            print('\n'"No convergence with residual", residualNorm, ">", tol, "in", maxiter, "iterations for time step", n)
            import sys
            sys.exit()

        # Here we have converged, so determine velocity and acceleration at time n+1
        velocityNew = np.multiply(a5, ua[free, 0]) + np.multiply(a6, displacementOld) + np.multiply(a7, velocityOld) + np.multiply(a8, accelerationOld)
        accelerationNew = np.multiply(a1, ua[free, 0]) + np.multiply(a2, displacementOld) + np.multiply(a3, velocityOld) + np.multiply(a4, accelerationOld)

        # Store the displacement at the tracked DOFs
        for i in range(numDispsToTrack):
            uTrack[i, n] = ua[trackedIndices[i], 0]
            vTrack[i, n] = velocityNew[trackedIndices[i]]
            aTrack[i, n] = accelerationNew[trackedIndices[i]]

        # Monitor degree of nonlinearity
        dnl1.append(firstOmega1/omega1ForDegreeOfNonlinearity)
        dnl2.append(firstOmega2/omega2ForDegreeOfNonlinearity)

        # Say something about the stiffness matrix
        determinant.append(np.linalg.det(Kf))
        gammasSay, omegasSay, vectorsSay = eigenInformation(Kf, 0, False, True)
        firstEigenvalueOfK.append(gammasSay[0])

        # Print the time
        if n>0:
            if (n % 10) == 0 or n == numTimePoints-1:
                if dt < 0.01:
                    print("%.3f (%.1f)  " % (t[n], dnl1[n]))
                else:
                    print("%.2f (%.1f)  " % (t[n], dnl1[n]))
            elif n==1:
                if dt < 0.01:
                    print('\n'"%.3f (%.1f)  " % (t[n], dnl1[n]), end="")
                else:
                    print('\n'"%.2f (%.1f)  " % (t[n], dnl1[n]), end="")
            else:
                if dt < 0.01:
                    print("%.3f (%.1f)  " % (t[n], dnl1[n]), end="")
                else:
                    print("%.2f (%.1f)  " % (t[n], dnl1[n]), end="")

        # DDM sensitivity calculations, prior to commit
        if len(DDMparameters) > 0:

            # Make sure the stiffness is up-to-date
            Ka[:] = 0.0
            for j in range(nelem):
                id, xyz, ug = model.localize(j, ua)
                element = elemlist[j]
                Fg_tilde, Kg = element.state(xyz, ug, 1)
                Ka[np.ix_(id, id)] = Ka[np.ix_(id, id)] + Kg
            Kf = Ka[np.ix_(free, free)]
            Keffective = Kf + a1 * Mf + a5 * Cf
            ddmKeffective = Keffective
            tangentNotAmended = True

            # Loop over DDM parameters
            for ddmIndex in range(len(DDMparameters)):

                # Reset for each parameter
                ddmRHSa = np.zeros(ntot)
                dFtildea = np.zeros(ntot)
                dKa = np.zeros((ntot, ntot))
                dKaInitial = np.zeros((ntot, ntot))
                dMa = np.zeros((ntot, ntot))
                dKadua = np.zeros((ntot, ntot, ntot))
                dcMdtheta = 0.0
                dcKdtheta = 0.0

                # Gather right-hand side contributions from elements
                if DDMparameters[ddmIndex][0] == 'Element':

                    # Check which elements have the parameter
                    ddmIsHere = np.full(nelem, False)
                    for eleNum in DDMparameters[ddmIndex][2]:
                        ddmIsHere[eleNum - 1] = True

                    # Must call all elements, even if the parameter isn't in them
                    for i in range(nelem):
                        id, xyz, ug = model.localize(i, ua)
                        element = elemlist[i]
                        dFtildeg, dKg, dKgdug = element.stateDerivative(xyz, ug, 1, DDMparameters[ddmIndex][1], ddmIndex, ddmIsHere[i], dampingModel[1])
                        dFtildea[id] = dFtildea[id] + dFtildeg
                        dKa[np.ix_(id, id)] = dKa[np.ix_(id, id)] + dKg
                        if not np.isscalar(dKgdug):
                            dKadua[np.ix_(id, id, id)] = dKadua[np.ix_(id, id, id)] + dKgdug

                    # Accommodate RTi
                    if dampingModel[0] == 'Rayleigh' and dampingModel[1] == 'Current' and dampingModel[2] == 'Initial':
                        for i in range(nelem):
                            id, xyz, ug = model.localize(i, ua)
                            element = elemlist[i]
                            dFtildeg, dKg, dKgdug = element.stateDerivative(xyz, ug, 1, DDMparameters[ddmIndex][1], ddmIndex, ddmIsHere[i], 'Initial')
                            dKaInitial[np.ix_(id, id)] = dKaInitial[np.ix_(id, id)] + dKg
                else:

                    if DDMparameters[ddmIndex][0] == 'Node' and DDMparameters[ddmIndex][1] == 'M':
                        dMa = model.getMassDerivative(DDMparameters[ddmIndex][2])
                        ddmRHSa -= np.multiply(np.diag(dMa), groundAcceleration[n])

                    elif DDMparameters[ddmIndex][0] == 'GroundMotion' and DDMparameters[ddmIndex][1] == 'Point':
                        if n == DDMparameters[ddmIndex][2]:
                            ddmRHSa[free] = ddmRHSa[free] + 9.81 * gmScaling * iotaMf

                    elif DDMparameters[ddmIndex][0] == 'Model' and DDMparameters[ddmIndex][1] == 'targetDamping':
                        if dampingModel[2] == 'Current' or dampingModel[2] == 'Initial':
                            dfactor = 2 / (omega1 + omega2)
                            dcMdtheta = omega1 * omega2 * dfactor
                            dcKdtheta = dfactor

                    elif DDMparameters[ddmIndex][0] == 'Model' and DDMparameters[ddmIndex][1] == 'cM':
                        dcMdtheta = 1.0
                        if dampingModel[2] == 'Current' or dampingModel[2] == 'Initial':
                            print('\n'"Error: cM or cK is a DDM parameter, but it's calculated, not given.")
                            import sys
                            sys.exit()

                    elif DDMparameters[ddmIndex][0] == 'Model' and DDMparameters[ddmIndex][1] == 'cK':
                        dcKdtheta = 1.0
                        if dampingModel[2] == 'Current' or dampingModel[2] == 'Initial':
                            print('\n'"Error: cM or cK is a DDM parameter, but it's calculated, not given.")
                            import sys
                            sys.exit()
                    else:
                        print('\n'"Error: Cannot handle DDM parameter of type", DDMparameters[ddmIndex][0])
                        import sys
                        sys.exit()

                    # Must call all elements, even if the parameter isn't in them
                    for i in range(nelem):
                        id, xyz, ug = model.localize(i, ua)
                        element = elemlist[i]
                        dFtildeg, dKg, dKgdug = element.stateDerivative(xyz, ug, 1, DDMparameters[ddmIndex][1], ddmIndex, False, dampingModel[1])
                        dFtildea[id] = dFtildea[id] + dFtildeg
                        dKa[np.ix_(id, id)] = dKa[np.ix_(id, id)] + dKg
                        dKadua[np.ix_(id, id, id)] = dKadua[np.ix_(id, id, id)] + dKgdug

                    # Accommodate RTi
                    if dampingModel[0] == 'Rayleigh' and dampingModel[1] == 'Current' and dampingModel[2] == 'Initial':
                        for i in range(nelem):
                            id, xyz, ug = model.localize(i, ua)
                            element = elemlist[i]
                            dFtildeg, dKg, dKgdug = element.stateDerivative(xyz, ug, 1, DDMparameters[ddmIndex][1], ddmIndex, False, 'Initial')
                            dKaInitial[np.ix_(id, id)] = dKaInitial[np.ix_(id, id)] + dKg

                # Pick up the derivative of mass and stiffness matrix from above
                dKf = dKa[np.ix_(free, free)]
                dMf = dMa[np.ix_(free, free)]

                # Accommodate RTi
                if dampingModel[0] == 'Rayleigh' and dampingModel[1] == 'Current' and dampingModel[2] == 'Initial':
                    dKfInitial = dKaInitial[np.ix_(free, free)]

                # dF/dtheta - dFtilde/dtheta
                ddmRHSf = ddmRHSa[free] - dFtildea[free]

                # (a2*M + a6*C)*dun/dtheta + (a3*M + a7*C)*dudotn/dtheta + (a4*M + a8*C)*dudotdotn/dtheta
                ddmRHSf -= Mf.dot(a2 * ddmDisplacementOld[:,ddmIndex] + a3 * ddmVelocityOld[:,ddmIndex] + a4 * ddmAccelerationOld[:,ddmIndex])
                ddmRHSf -= Cf.dot(a6 * ddmDisplacementOld[:,ddmIndex] + a7 * ddmVelocityOld[:,ddmIndex] + a8 * ddmAccelerationOld[:,ddmIndex])

                # dM/dtheta * (a1*un1 + a2*un + a3*udotn + a4*udotdotn)
                ddmRHSf -= dMf.dot(a1 * ua[free, 0] + a2 * displacementOld + a3 * velocityOld + a4 * accelerationOld)

                # Store the parenthesis that multiplies with the derivative of the damping matrix
                a5parenthesis = a5 * ua[free, 0] + a6 * displacementOld + a7 * velocityOld + a8 * accelerationOld

                # Derivatives for dC/dtheta * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                if dampingModel[0] == 'Rayleigh':

                    # cM * dM/dtheta * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                    ddmRHSf -= cM * dMf.dot(a5parenthesis)

                    # cK * dK/dtheta * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                    ddmRHSf -= cK * dKf.dot(a5parenthesis)

                    # dcM/dtheta*M*(a5*un1+a6*un+a7*udotn+a8*udotdotn) if cM is the DDM parameter
                    ddmRHSf -= dcMdtheta * Mf.dot(a5parenthesis)

                    # dcK/dtheta*K*(a5*un1+a6*un+a7*udotn+a8*udotdotn) if cK is the DDM parameter
                    ddmRHSf -= dcKdtheta * KfDamping.dot(a5parenthesis)

                    if dampingModel[2] == 'Current' or dampingModel[2] == 'Initial':

                        # Accommodate RTi
                        if dampingModel[1] == 'Current' and dampingModel[2] == 'Initial':
                            dKf = dKfInitial

                        # domega/dtheta = domega/dgamma * dgamma/dtheta, for fixed displacement, for two relevant frequencies
                        lhs = (vectorsDamping[dampingModel[3]-1].dot(Mf)).dot(vectorsDamping[dampingModel[3]-1])
                        rhs = (vectorsDamping[dampingModel[3]-1].dot(dKf)).dot(vectorsDamping[dampingModel[3]-1]) - gammasDamping[dampingModel[3]-1] * (vectorsDamping[dampingModel[3]-1].dot(dMf)).dot(vectorsDamping[dampingModel[3]-1])
                        dgamma1dtheta = rhs/lhs
                        domega1 = 0.5 / np.sqrt(gammasDamping[dampingModel[3]-1]) * dgamma1dtheta

                        lhs = (vectorsDamping[dampingModel[4]-1].dot(Mf)).dot(vectorsDamping[dampingModel[4]-1])
                        rhs = (vectorsDamping[dampingModel[4]-1].dot(dKf)).dot(vectorsDamping[dampingModel[4]-1]) - gammasDamping[dampingModel[4]-1] * (vectorsDamping[dampingModel[4]-1].dot(dMf)).dot(vectorsDamping[dampingModel[4]-1])
                        dgamma2dtheta = rhs/lhs
                        domega2 = 0.5 / np.sqrt(gammasDamping[dampingModel[4]-1]) * dgamma2dtheta

                        # Derivatives dc/dtheta = dc/domega * domega/dtheta
                        dfactor = -2 * dampingModel[5] / (omega1 + omega2)**2 * (domega1 + domega2)
                        dcM = domega1 * omega2 * factor + omega1 * domega2 * factor + omega1 * omega2 * dfactor
                        dcK = dfactor

                        # dcM/dtheta * M * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                        ddmRHSf -= dcM * Mf.dot(a5parenthesis)

                        # dcM/dtheta * M * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                        ddmRHSf -= dcK * KfDamping.dot(a5parenthesis)

                        # Amend effective stiffness with derivative of cM(u) and cK(u) if the Rayleigh damping uses current stiffness
                        if dampingModel[1] == 'Current' and dampingModel[2] == 'Current' and tangentNotAmended:

                            # dc/domega
                            dcMdomega1 = omega2 * 2 * dampingModel[5] / (omega1+omega2) - omega1 * omega2 * 2 * dampingModel[5] / (omega1+omega2)**2
                            dcMdomega2 = omega1 * 2 * dampingModel[5] / (omega1+omega2) - omega1 * omega2 * 2 * dampingModel[5] / (omega1+omega2)**2
                            dcKdomega1 = -2 * dampingModel[5] / (omega1+omega2)**2
                            dcKdomega2 = -2 * dampingModel[5] / (omega1+omega2)**2

                            # domega/dgamma
                            domega1dgamma1 = 0.5 / np.sqrt(gammas[dampingModel[3]-1])
                            domega2dgamma2 = 0.5 / np.sqrt(gammas[dampingModel[4]-1])

                            # dgamma/dK
                            lhs = (vectors[dampingModel[3]-1].dot(Mf)).dot(vectors[dampingModel[3]-1])
                            rhs = np.outer(vectors[dampingModel[3]-1], vectors[dampingModel[3]-1])
                            dgamma1dK = rhs / lhs

                            lhs = (vectors[dampingModel[4]-1].dot(Mf)).dot(vectors[dampingModel[4]-1])
                            rhs = np.outer(vectors[dampingModel[4]-1], vectors[dampingModel[4]-1])
                            dgamma2dK = rhs / lhs

                            # dK/du
                            dKfduf = dKadua[np.ix_(free, free, free)]

                            # dcdu
                            dcMdu = np.zeros(len(free))
                            dcKdu = np.zeros(len(free))
                            for m in range(len(free)):
                                for i in range(len(free)):
                                    for j in range(len(free)):
                                        dcMdu[m] += (dcMdomega1*domega1dgamma1*dgamma1dK[i,j]+dcMdomega2*domega2dgamma2*dgamma2dK[i,j])*dKfduf[i,j,m]
                                        dcKdu[m] += (dcKdomega1*domega1dgamma1*dgamma1dK[i,j]+dcKdomega2*domega2dgamma2*dgamma2dK[i,j])*dKfduf[i,j,m]

                            # Ma5
                            Ma5 = Mf.dot(a5parenthesis)
                            Ka5 = Kf.dot(a5parenthesis)

                            # Amendment of effective stiffness
                            Kamendment = np.zeros((len(free), len(free)))
                            for m in range(len(free)):
                                for l in range(len(free)):
                                    Kamendment[l,m] = dcMdu[m] * Ma5[l] + dcKdu[m] * Ka5[l]

                            ddmKeffective = ddmKeffective + Kamendment

                    # Amend effective stiffness if the Rayleigh damping uses current stiffness
                    if dampingModel[1] == 'Current' and tangentNotAmended:

                        tangentNotAmended = False

                        # Amend effective stiffness with derivative of current tangent stiffness
                        dKfduf = dKadua[np.ix_(free, free, free)]
                        dKfcontracted = np.tensordot(dKfduf, a5parenthesis, axes=([1],[0]))

                        ddmKeffective = ddmKeffective + cK * dKfcontracted

                elif dampingModel[0] == 'Modal':

                    # Derivative of eigenvalues and eigenvectors wrt. theta
                    dgammasdtheta, domegasdtheta, dvectorsdtheta = eigenDerivativesWrtTheta(KfDamping, dKf, Mf, dMf, gammasDamping, vectorsDamping)

                    # dC/dtheta (here we have to be careful to use the information that was used to calculate the damping matrix)
                    term1 = np.zeros((len(free), len(free)))
                    term2 = np.zeros((len(free), len(free)))
                    term3 = np.zeros((len(free), len(free)))
                    term4 = np.zeros((len(free), len(free)))
                    term5 = np.zeros((len(free), len(free)))
                    term6 = np.zeros((len(free), len(free)))
                    for i in range(len(gammas)):
                        modalMass = (vectorsDamping[i].dot(Mf)).dot(vectorsDamping[i])
                        dmodalmass = (dvectorsdtheta[i].dot(Mf)).dot(vectorsDamping[i]) + \
                                     (vectorsDamping[i].dot(dMf)).dot(vectorsDamping[i]) + \
                                     (vectorsDamping[i].dot(Mf)).dot(dvectorsdtheta[i])
                        outer1 = np.outer(vectorsDamping[i], vectorsDamping[i])
                        outer2 = np.outer(dvectorsdtheta[i], vectorsDamping[i])
                        outer3 = np.outer(vectorsDamping[i], dvectorsdtheta[i])
                        term1 = term1 + 2.0 * dampingModel[2] * omegasDamping[i] / modalMass * outer1
                        term2 = term2 + 2.0 * dampingModel[2] * 0.5 / omegasDamping[i] * dgammasdtheta[i] / modalMass * outer1
                        term3 = term3 + 2.0 * dampingModel[2] * omegasDamping[i] / modalMass**2 * dmodalmass * outer1
                        term4 = term4 + 2.0 * dampingModel[2] * omegasDamping[i] / modalMass * outer2
                        term5 = term5 + 2.0 * dampingModel[2] * omegasDamping[i] / modalMass * outer3
                        term6 = term6 + 2.0 * dampingModel[2] * omegasDamping[i] / modalMass * outer1

                    dCf = (dMf.dot(term1)).dot(Mf)
                    dCf += (Mf.dot(term2)).dot(Mf)
                    dCf -= (Mf.dot(term3)).dot(Mf)
                    dCf += (Mf.dot(term4)).dot(Mf)
                    dCf += (Mf.dot(term5)).dot(Mf)
                    dCf += (Mf.dot(term6)).dot(dMf)

                    # Add contribution from target damping being theta
                    if DDMparameters[ddmIndex][1] == 'targetDamping':
                        dCf += np.multiply(Cf, 1.0/dampingModel[2])

                    # dC/dtheta * (a5*un1 + a6*un + a7*udotn + a8*udotdotn)
                    ddmRHSf -= dCf.dot(a5parenthesis)

                    # Amend effective stiffness with derivative of omega if modal damping uses current stiffness
                    if dampingModel[1] == 'Current' and tangentNotAmended:

                        tangentNotAmended = False

                        # Derivative of eigenvalues and eigenvectors wrt. K
                        dgammasdK, domegasdK, dvectorsdK = eigenDerivativesWrtK(KfDamping, Mf, gammasDamping, vectorsDamping)

                        # Pick up ingredients
                        zeta = dampingModel[2]
                        dKfduf = dKadua[np.ix_(free, free, free)]

                        # Amend coefficient matrix
                        term1 = 0.0
                        term2 = 0.0
                        term3 = 0.0
                        term4 = 0.0
                        for mode in range(len(gammas)):
                            m = (vectorsDamping[mode].dot(Mf)).dot(vectorsDamping[mode])
                            b = Mf.dot(vectorsDamping[mode])
                            term1 -= 4*zeta*omegasDamping[mode]/m**2 * np.einsum('i,k,q,qop->ikop', b, b, b, dvectorsdK[mode])
                            term2 += 2 * zeta / m * np.einsum('i,k,op->ikop', b, b, domegasdK[mode])
                            term3 += 2*zeta*omegasDamping[mode]/m * np.einsum('iq,k,qop->ikop', Mf, b, dvectorsdK[mode])
                            term4 += 2*zeta*omegasDamping[mode]/m * np.einsum('i,qk,qop->ikop', b, Mf, dvectorsdK[mode])
                        terms = term1 + term2 + term3 + term4
                        amendment = np.einsum('ikop,opr,k->ir', terms, dKfduf, a5parenthesis)
                        ddmKeffective = ddmKeffective + amendment

                else:
                    print('\n'"Error: Cannot understand the damping model")
                    import sys
                    sys.exit()

                # Solve for the derivative of the displacements
                ddmDisplacementNew = np.linalg.solve(ddmKeffective, ddmRHSf)

                # Update the derivatives of velocity and acceleration responses
                ddmAccelerationNew = np.multiply(a1, ddmDisplacementNew) + np.multiply(a2, ddmDisplacementOld[:,ddmIndex]) + np.multiply(a3, ddmVelocityOld[:,ddmIndex]) + np.multiply(a4, ddmAccelerationOld[:,ddmIndex])
                ddmVelocityNew = np.multiply(a5, ddmDisplacementNew) + np.multiply(a6, ddmDisplacementOld[:,ddmIndex]) + np.multiply(a7, ddmVelocityOld[:,ddmIndex]) + np.multiply(a8, ddmAccelerationOld[:,ddmIndex])

                # Now that we're done; reset the sensitivity vectors at the previous step
                ddmDisplacementOld[:,ddmIndex] = ddmDisplacementNew
                ddmVelocityOld[:,ddmIndex] = ddmVelocityNew
                ddmAccelerationOld[:,ddmIndex] = ddmAccelerationNew

                # Pick up the derivative that will be plotted
                ddmua[free] = ddmDisplacementNew
                for i in range(numDispsToTrack):
                    dudx[i, ddmIndex, n] = ddmua[trackedIndices[i]]
                    dvdx[i, ddmIndex, n] = ddmVelocityNew[trackedIndices[i]]
                    dadx[i, ddmIndex, n] = ddmAccelerationNew[trackedIndices[i]]

                # All materials must store unconditional derivatives for all ddm variables
                for i in range(nelem):
                    id, xyz, ug = model.localize(i, ua)
                    id, xyz, ddmug = model.localize(i, ddmua)
                    element = elemlist[i]
                    if DDMparameters[ddmIndex][0] == 'Element':
                        element.commitSensitivity(xyz, ug, ddmug, DDMparameters[ddmIndex][1], ddmIndex, ddmIsHere[i])
                    else:
                        element.commitSensitivity(xyz, ug, ddmug, DDMparameters[ddmIndex][1], ddmIndex, False)

        # Commit and update element states using converged displacements
        for i in range(nelem):
            id, xyz, ug = model.localize(i, ua)
            element = elemlist[i]
            element.commit(xyz, ug)

        # Set response values at time step i equal to response at time n+1
        displacementOld = ua[free, 0]
        velocityOld = velocityNew
        accelerationOld = accelerationNew

        # Zero second column of displacement vector after commit
        ua[:, 1] = 0.0

    # ------------------------------------------------------------------------
    # PLOT THE RESPONSE & RETURN
    # ------------------------------------------------------------------------

    if plotFlag:
        plt.ion()
        plt.figure()
        plt.autoscale(True)
        plt.title("Displacement (max=%.3f, min=%.3f)" % (np.max(uTrack[0,:]), np.min(uTrack[0,:])))
        for i in range(len(t)-1):
            if dnl1[i] > 4:
                plt.plot([t[i], t[i+1]], [uTrack[0,i], uTrack[0,i+1]], 'r-')
            elif dnl1[i] > 2:
                    plt.plot([t[i], t[i + 1]], [uTrack[0,i], uTrack[0,i+1]], 'm-')
            elif dnl1[i] > 1.01:
                plt.plot([t[i], t[i+1]], [uTrack[0,i], uTrack[0,i+1]], 'b-')
            else:
                plt.plot([t[i], t[i+1]], [uTrack[0,i], uTrack[0,i+1]], 'k-')
        plt.xlabel("Time")
        plt.ylabel("Displacement")
        print('\n''\n'"Maximum degree of nonlinearity: %.2f" % np.max(dnl1))
        print('\n'"Click somewhere in the plot to continue...")
        plt.waitforbuttonpress()

    # Tracked displacements: [which-u, u-values]     Tracked DDMs: [which-u, which-x, dudx-values]
    return t, groundAcceleration, uTrack, vTrack, aTrack, dudx, dvdx, dadx, dnl1, dnl2