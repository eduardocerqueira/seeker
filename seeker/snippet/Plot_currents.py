#date: 2025-02-04T17:10:33Z
#url: https://api.github.com/gists/5f2c7bf8062fa7d5e820ee1729859c2b
#owner: https://api.github.com/users/stefaniasif

import numpy as np
import matplotlib.pyplot as plt
import initialize_state_vector as isv
import Load_parameters as lp


def Plot_currents(t,x):

    # theta
    theta = lp.omega_0 * t

    inv_LD_mat = np.linalg.inv(lp.LD_mat)
    inv_LQ_mat = np.linalg.inv(lp.LQ_mat)

    # d-axis currents,  e = L*i,    i = inv.L * e
    x_d = np.vstack((x[0,:], x[2,:], x[3,:]))
    i_d_fd_kd = inv_LD_mat.dot(x_d)
    
    # q-axis currents
    x_q = np.vstack((x[1,:], x[4,:]))
    i_q_kq = inv_LQ_mat.dot(x_q)


    # Stator phase currents
    
    #dq0_inverse = np.array([
    #    [np.cos(theta), -np.sin(theta), 1],
    #    [np.cos(theta - 2*np.pi/3), -np.sin(theta - 2*np.pi/3), 1],
    #    [np.cos(theta + 2*np.pi/3), -np.sin(theta + 2*np.pi/3), 1]
    #])

    #i_dq0 = np.vstack((i_d, i_q, i_0))
    #i_abc = dq0_inverse.dot(i_dq0)
    
    # abc stator phase currents after inverse dq0-transformation, i0 = 0 for three phase fault
    i_a = np.cos(theta) * i_d_fd_kd[0,:] - np.sin(theta) * i_q_kq[0,:]
    i_b = np.cos(theta - 2*np.pi/3) * i_d_fd_kd[0,:] - np.sin(theta - 2*np.pi/3) * i_q_kq[0,:]
    i_c = np.cos(theta + 2*np.pi/3) * i_d_fd_kd[0,:] - np.sin(theta + 2*np.pi/3) * i_q_kq[0,:]





    # Plot results
    plt.figure(figsize=(10, 6))

    # Plot field winding current
    plt.subplot(3, 1, 1)
    plt.plot(t, i_d_fd_kd[1,:], label=r'$i_{fd}$', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pu)')
    plt.title('Field Winding Current')
    plt.legend()
    plt.grid()

    ## Plot abc stator phase currents
    plt.subplot(3, 1, 2)
    plt.plot(t, i_a, label=r'$i_a$', color='r')
    plt.plot(t, i_b, label=r'$i_b$', color='g')
    plt.plot(t, i_c, label=r'$i_c$', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pu)')
    plt.title('Stator Phase Currents')
    plt.legend()
    plt.grid()

    # Plot damper winding currents
    plt.subplot(3, 1, 3)
    plt.plot(t, i_d_fd_kd[2,:], label=r'$i_{kd}$', color='purple')
    plt.plot(t, i_q_kq[1,:], label=r'$i_{kq}$', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pu)')
    plt.title('Damper Winding Currents')
    plt.legend()
    plt.grid()

    #plt.tight_layout()
    plt.show()
