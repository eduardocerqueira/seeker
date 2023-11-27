#date: 2023-11-27T17:09:20Z
#url: https://api.github.com/gists/8ecbd02e3e318a979189c41fa05c5edd
#owner: https://api.github.com/users/xsuz

import numpy as np
import matplotlib.pyplot as plt

def analysis_foil(z,alpha=0):
    N=len(z)-1
    U=np.exp(1j*np.pi/180*alpha) # freestream velocity
    # 1. Calculate gamma
    #
    # First, vortex sheet is assumed to be placed on the panels.
    # Then, Solve the linear system to satisfy the following conditions.
    #
    # Boundary Conditions:
    #   1. The velocity of the component orthogonal to the panel is 0
    #   2. Kutta condition
    #
    # Linear system:
    #   A*gamma=B
    #   A: N+1 x N+1 matrix
    #   gamma: N+1 vector
    #   B: N+1 vector

    A=np.zeros((N+1,N+1))
    B=np.zeros(N+1)
    S1=np.zeros((N,N),dtype=complex)
    S2=np.zeros((N,N),dtype=complex)

    l=np.zeros(N)
    n=np.zeros(N,dtype=complex)
    z_ref=np.zeros(N,dtype=complex)

    for k in range(N):
        l[k]=np.abs(z[k+1]-z[k])
        z_ref[k]=(z[k+1]+z[k])/2
        n[k]=(z[k+1]-z[k])/(l[k]*1j)

    for k in range(N):
        for j in range(N):
            S1[k][j]=((1j*l[j])/(2*np.pi*(z[j+1]-z[j])))*(-1+(z[j+1]-z_ref[k])/(z[j+1]-z[j])*np.log((z[j+1]-z_ref[k])/(z[j]-z_ref[k])))
            S2[k][j]=((1j*l[j])/(2*np.pi*(z[j+1]-z[j])))*(+1-(z[j]-z_ref[k])/(z[j+1]-z[j])*np.log((z[j+1]-z_ref[k])/(z[j]-z_ref[k])))

    for k in range(N):
        for j in range(N+1):
            if j==0:
                A[k][j]=np.real(S1[k][j]*n[k])
            elif j==N:
                A[k,j]=np.real(S2[k][j-1]*n[k])
            else:
                A[k,j]=np.real((S1[k][j]+S2[k,j-1])*n[k])

    # Kutta condition
    A[N,0]=1
    A[N,N]=1

    for k in range(N):
        B[k]=-np.real(np.conj(U)*n[k])
    # Kutta condition
    B[N]=0

    gamma=np.linalg.solve(A,B) # solve the linear system

    S1_uv=np.zeros(N,dtype=complex)
    S2_uv=np.zeros(N,dtype=complex)

    def calc_uv(z_arg):
        for k in range(N):
            S1_uv[k]=((1j*l[k])/(2*np.pi*(z[k+1]-z[k])))*(-1+(z[k+1]-z_arg)/(z[k+1]-z[k])*np.log((z[k+1]-z_arg)/(z[k]-z_arg)))
            S2_uv[k]=((1j*l[k])/(2*np.pi*(z[k+1]-z[k])))*(+1-(z[k]-z_arg)/(z[k+1]-z[k])*np.log((z[k+1]-z_arg)/(z[k]-z_arg)))
        u=np.conj(U)+S1_uv[0]*gamma[0]+S2_uv[N-1]*gamma[N]
        for k in range(1,N):
            u+=S1_uv[k]*gamma[k]+S2_uv[k-1]*gamma[k]
        return np.conj(u)

    # 2. Calculate Cl and Cm
    Cp=np.zeros(N)
    z_ref*=1+1e-14
    u_ref=np.zeros(N,dtype=complex)
    for k in range(N):
        u_ref[k]=calc_uv(z_ref[k])
        Cp[k]=1-np.abs(u_ref[k]/U)**2

    Cl=0
    Cm=0
    for k in range(N):
        theta=np.angle(z[k+1]-z[k])
        pn=Cp[k]*l[k]*np.cos(theta)
        pt=-Cp[k]*l[k]*np.sin(theta)
        Cl+=Cp[k]*l[k]*np.cos(alpha/180*np.pi-theta)
        Cm+=-pn*np.real(z_ref[k]-0.25)+pt*np.imag(z_ref[k])
    return calc_uv,Cl,Cm


if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        airfoil_file=sys.argv[1]
    else:
        airfoil_file=input("Airfoil data file: ")
    if len(sys.argv)>2:
        alfa=float(sys.argv[2])
    else:
        alfa=float(input("Angle of attack: "))
    # Read airfoil data
    z=[]
    with open(airfoil_file) as f:
        lines = f.readlines()
        for line in lines[1:]:
            z.append(float(line.split()[0])+float(line.split()[1])*1j)
    z=np.array(z)
    calc_uv,Cl,Cm=analysis_foil(z,alfa)
    print(f"{Cl=}, {Cm=}")
    NUM_PANEL=70
    xx,yy=np.meshgrid(np.linspace(-1.5,1.5,NUM_PANEL),np.linspace(-1.5,1.5,NUM_PANEL))
    # To plot the airfoil, the coordinates are rotated by alfa degrees.
    z_calc=(xx+yy*1j)*np.exp(1j*np.pi/180*alfa)
    u_calc=np.zeros((NUM_PANEL,NUM_PANEL),dtype=complex)
    cp_calc=np.zeros((NUM_PANEL,NUM_PANEL))
    for k in range(NUM_PANEL):
        for j in range(NUM_PANEL):
            u_calc[k,j]=calc_uv(z_calc[k,j])
            cp_calc[k,j]=1-np.abs(u_calc[k,j])**2
    u_calc*=np.exp(-1j*np.pi/180*alfa)
    z*=np.exp(-1j*np.pi/180*alfa)
    z_calc*=np.exp(-1j*np.pi/180*alfa)

    fig, ax = plt.subplots()
    plt.title(f"alpha={alfa}")
    cs=ax.contourf(xx,yy,cp_calc,levels=NUM_PANEL,cmap='jet')
    ax.quiver(np.real(z_calc),np.imag(z_calc),np.real(u_calc),np.imag(u_calc),linewidth=0.01,scale=100)
    ax.plot(np.real(z),np.imag(z))
    fig.colorbar(cs)
    plt.xlim(-0.5,1.5)
    plt.ylim(-1,1)
    plt.show()