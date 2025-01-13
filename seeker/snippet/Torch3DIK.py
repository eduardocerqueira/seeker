#date: 2025-01-13T17:10:19Z
#url: https://api.github.com/gists/f0d773b4e4cb3ee97b61142dc317ca70
#owner: https://api.github.com/users/ctralie

import torch
import numpy as np
import matplotlib.pyplot as plt

offsets = [torch.zeros(3), torch.tensor([1, 0, 0]), torch.tensor([1, 0, 0]), torch.tensor([1, 0, 0])]
Rs = [torch.eye(3)[:, 0:2] for _ in range(len(offsets))]
for i in range(len(Rs)-1):
    Rs[i] = Rs[i].requires_grad_()

def get_positions(offsets, Rs):
    ## Step 1: Fix up the rotation matrices
    with torch.no_grad():
        for R in Rs:
            ## Step 1a: Normalize the two columns
            R /= torch.sqrt(torch.sum(R**2, dim=0, keepdims=True))
            ## Step 1b: Do one step of Gram-Schmidt to make 
            ## second column perpendicular to the first
            ## Subtract off projection of second column onto first column
            R[:, 1] -= torch.sum(R[:, 0]*R[:, 1])*R[:, 0]
            R[:, 1] /= torch.sqrt(torch.sum(R[:, 1]**2))
    
    ## Step 2: Setup the entire transformation for each joint
    Ms = [torch.eye(4) for _ in range(len(Rs))]
    for M, R, offset in zip(Ms, Rs, offsets):
        M[0:3, 0:2] = R
        M[0:3, 2] = torch.cross(R[:, 0], R[:, 1], dim=0)
        M[0:3, 3] = offset
    
    ## Step 3: Apply these matrices in a hierarchy to get the
    ## final positions of the joints in world coordinates
    MCurr = torch.eye(4)
    positions = []
    for M in Ms:
        MCurr = torch.matmul(MCurr, M)
        positions.append(MCurr[0:3, 3])
    
    return positions


target = torch.tensor([1, 1, 1])


plt.ion()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_proj_type('ortho')
ax.scatter(target[0], target[1], target[2], color='red')
P = get_positions(offsets, Rs)
P = np.array([p.detach().cpu().numpy() for p in P])
lines,  = ax.plot(P[:, 0], P[:, 1], P[:, 2])
points = ax.scatter(P[:, 0], P[:, 1], P[:, 2])

optimizer = torch.optim.Adam(Rs, lr=1e-3)
n_iters = 10000
losses = []
for i in range(n_iters):
    ## Step 1: Clear accumulation of gradients from previous steps
    optimizer.zero_grad()
    ## Step 2: Compute loss function
    positions = get_positions(offsets, Rs)
    loss = torch.sum((positions[-1] - target)**2)
    losses.append(loss.item())
    ## Step 3: Compute gradient of loss function wrt all parameters in the optimizer
    loss.backward() 
    ## Step 4: Take a step of the optimizer against the gradient to minimize against loss
    optimizer.step()

    if i%10 == 0:
        P = get_positions(offsets, Rs)
        P = np.array([p.detach().cpu().numpy() for p in P])

        dP = P[1:, :] - P[0:-1, :]
        print(np.sum(dP**2, axis=1))

        lines.set_data_3d(*(P.T))
        points._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        fig.canvas.draw()
        fig.canvas.flush_events()

