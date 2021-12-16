#date: 2021-12-16T17:00:54Z
#url: https://api.github.com/gists/4c25e9d266f9762259606349594b802a
#owner: https://api.github.com/users/jkwashbourne

import numpy as np
import socket
from devito.mpi import MPI
from sympy import sqrt, sin, cos
from devito import (Grid, Function, TimeFunction, Eq, Operator, norm)
from examples.seismic import RickerSource, TimeAxis

import example_cmdline
nt,nx,ny,nz,block = example_cmdline.parse()
print("nt,nx,ny,nz; %3d %3d %3d %3d" % (nt,nx,ny,nz))

space_order = 8
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 250.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

shape = (nx, ny, nz)
spacing = (10.0, 10.0, 10.0)
origin = tuple([0.0 for s in shape])
extent = tuple([d * (s - 1) for s, d in zip(shape, spacing)])
grid = Grid(extent=extent, shape=shape, origin=origin, dtype=dtype)

b = Function(name='b', grid=grid, space_order=space_order)
f = Function(name='f', grid=grid, space_order=space_order)
vel = Function(name='vel', grid=grid, space_order=space_order)
eps = Function(name='eps', grid=grid, space_order=space_order)
eta = Function(name='eta', grid=grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=grid, space_order=space_order)
theta = Function(name='theta', grid=grid, space_order=space_order)
phi = Function(name='phi', grid=grid, space_order=space_order)

b.data[:] = 1.0
f.data[:] = 0.84
vel.data[:] = 1.5
eps.data[:] = 0.2
eta.data[:] = 0.4
wOverQ.data[:] = 1.0
theta.data[:] = np.pi / 3
phi.data[:] = np.pi / 6

t0 = 0.0
t1 = 1.0 * nt
dt = 1.0
time_axis = TimeAxis(start=t0, stop=t1, step=dt)

p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=space_order)
t, x, y, z = p0.dimensions

src_coords = np.empty((1, len(shape)), dtype=dtype)
src_coords[0, :] = [d * (s-1)//2 for d, s in zip(spacing, shape)]
src = RickerSource(name='src', grid=grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]
src_term = src.inject(field=p0.forward, expr=src * t.spacing**2 * vel**2 / b)


def g1(field):
    return (cos(theta) * cos(phi) * field.dx(x0=x+x.spacing/2) +
            cos(theta) * sin(phi) * field.dy(x0=y+y.spacing/2) -
            sin(theta) * field.dz(x0=z+z.spacing/2))


def g2(field):
    return - (sin(phi) * field.dx(x0=x+x.spacing/2) -
              cos(phi) * field.dy(x0=y+y.spacing/2))


def g3(field):
    return (sin(theta) * cos(phi) * field.dx(x0=x+x.spacing/2) +
            sin(theta) * sin(phi) * field.dy(x0=y+y.spacing/2) +
            cos(theta) * field.dz(x0=z+z.spacing/2))


def g1_tilde(field):
    return ((cos(theta) * cos(phi) * field).dx(x0=x-x.spacing/2) +
            (cos(theta) * sin(phi) * field).dy(x0=y-y.spacing/2) -
            (sin(theta) * field).dz(x0=z-z.spacing/2))


def g2_tilde(field):
    return - ((sin(phi) * field).dx(x0=x-x.spacing/2) -
              (cos(phi) * field).dy(x0=y-y.spacing/2))


def g3_tilde(field):
    return ((sin(theta) * cos(phi) * field).dx(x0=x-x.spacing/2) +
            (sin(theta) * sin(phi) * field).dy(x0=y-y.spacing/2) +
            (cos(theta) * field).dz(x0=z-z.spacing/2))


# Time update equation for quasi-P state variable p
update_p = t.spacing**2 * vel**2 / b * \
    (g1_tilde(b * (1 + 2 * eps) * g1(p0)) +
     g2_tilde(b * (1 + 2 * eps) * g2(p0)) +
     g3_tilde(b * (1 - f * eta**2) * g3(p0) + b * f * eta * sqrt(1 - eta**2) * g3(m0))) + \
    (2 - t.spacing * wOverQ) * p0 + \
    (t.spacing * wOverQ - 1) * p0.backward

# Time update equation for quasi-S state variable m
update_m = t.spacing**2 * vel**2 / b * \
    (g1_tilde(b * (1 - f) * g1(m0)) +
     g2_tilde(b * (1 - f) * g2(m0)) +
     g3_tilde(b * (1 - f + f * eta**2) * g3(m0) + b * f * eta * sqrt(1 - eta**2) * g3(p0))) + \
    (2 - t.spacing * wOverQ) * m0 + \
    (t.spacing * wOverQ - 1) * m0.backward

stencil_p = Eq(p0.forward, update_p)
stencil_m = Eq(m0.forward, update_m)

dt = time_axis.step
spacing_map = grid.spacing_map
spacing_map.update({t.spacing: dt})

op = Operator([stencil_p, stencil_m, src_term], subs=spacing_map, name='OpExampleTti')
op.apply()
print("norm(p0); ", norm(p0))

# print(op.args)
# f = open("operator.tti.c", "w")
# print(op, file=f)
# f.close()

# if block == 0:
#     bx = 32
#     by = 15
#     op.apply(x0_blk0_size=bx, y0_blk0_size=by)

#     print("")
#     print(time_axis)
#     print("nx,ny,nz,norm; %5d %5d %5d %12.6e" % (shape[0], shape[1], shape[2], norm(p0)))

# else:
#     filename = "timing_tti.%s.txt" % (socket.gethostname())
#     print("filename; ", filename)

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()

#     bx1 = 2
#     bx2 = 32
#     dbx = 1
#     by1 = 2
#     by2 = 32
#     dby = 1

#     f = open(filename, "w")

#     if rank == 0:
#         print("nt,nx,ny,nz; %5d %5d %5d %5d" % (nt, nx, ny, nz), file=f)

#     for bx in range(bx2, bx1-1, -dbx):
#         for by in range(by2, by1-1, -dby):
#             p0.data[:] = 0
#             s = op.apply(x0_blk0_size=bx, y0_blk0_size=by)
#             normp0 = norm(p0)
#             if rank == 0:
#                 gpointss = np.sum([v.gpointss for k, v in s.items()])
#                 print("bx,by,gpts/s; %3d %3d %10.6f %12.8f" % (bx, by, gpointss, normp0))
#                 print("bx,by,gpts/s; %3d %3d %10.6f %12.8f" % (bx, by, gpointss, normp0), file=f)
#                 f.flush()

#     f.close()
