#date: 2024-05-23T17:10:26Z
#url: https://api.github.com/gists/c3c685a8b2ec1f14171772bd7bc7ea3e
#owner: https://api.github.com/users/metab0t

import pandas as pd
import numpy as np

import time

from pyoframe import Model, Variable, sum
import gurobipy as gp

import pyoptinterface as poi
from pyoptinterface import gurobi

np.random.seed(1234567)


def gen_input(Nplants, Nhouses):
    plants = (
        pd.DataFrame(
            dict(
                plant=range(Nplants),
                capacity=np.random.uniform(0.0, 100.0, Nplants),
                fixed_cost=np.random.uniform(0.0, 100.0, Nplants),
            )
        )
        .astype({"plant": "int64"})
        .set_index("plant")
    )

    wharehouses = (
        pd.DataFrame(
            dict(
                wharehouse=range(Nhouses),
                demand=np.random.uniform(0.0, 100.0, Nhouses),
            )
        )
        .astype({"wharehouse": "int64"})
        .set_index("wharehouse")
    )

    transport_costs = (
        pd.DataFrame(
            dict(
                wharehouse=np.repeat(range(Nhouses), Nplants),
                # 0...0(Nhouses) 1...1 (Nhouses) ... (Nplants-1)...(Nplants-1) (Nhouses)
                # use numpy
                plant=np.tile(range(Nplants), Nhouses),
                cost=np.random.uniform(0.0, 100.0, Nhouses * Nplants),
            )
        )
        .astype({"plant": "int64", "wharehouse": "int64"})
        .set_index(["wharehouse", "plant"])["cost"]
    )

    transport_costs_numpy = transport_costs.to_numpy().reshape(Nhouses, Nplants)

    return plants, wharehouses, transport_costs, transport_costs_numpy


def pyoframe_main(plants, warehouses, transport_costs):
    t0 = time.time()
    m = Model("min")
    m.open = Variable(plants.index, vtype="binary")
    m.transport = Variable(warehouses.index, plants.index, lb=0)

    m.con_max_capacity = sum("wharehouse", m.transport) <= plants.capacity * m.open
    m.con_meet_demand = sum("plant", m.transport) == warehouses.demand

    m.objective = sum(m.open * plants.fixed_cost) + sum(m.transport * transport_costs)

    m.params.Method = 2
    m.to_file("model.lp", use_var_names=True)

    gm = gp.read("model.lp")

    t1 = time.time()
    print(f"Pyoframe elapsed time: {t1-t0:.2f} seconds")

    return m


def add_ndarray_variable(m, shape, **kwargs):
    array = np.empty(shape, dtype=object)
    array_flat = array.flat
    for i in range(array.size):
        array_flat[i] = m.add_variable(**kwargs)
    return array


def poi_main(plants, warehouses, transport_costs_numpy):
    demand = warehouses["demand"].values
    capacity = plants["capacity"].values
    fixedCosts = plants["fixed_cost"].values

    plants = range(len(capacity))
    warehouses = range(len(demand))

    t0 = time.time()
    m = gurobi.Model()

    open = add_ndarray_variable(m, len(plants), domain=poi.VariableDomain.Binary)
    transport = add_ndarray_variable(m, (len(warehouses), len(plants)), lb=0)

    for p in plants:
        expr = poi.quicksum(transport[:, p])
        expr -= capacity[p] * open[p]
        m.add_linear_constraint(expr, poi.Leq, 0.0)

    for w in warehouses:
        expr = poi.quicksum(transport[w])
        expr -= demand[w]
        m.add_linear_constraint(expr, poi.Eq, 0.0)

    obj = poi.ExprBuilder()
    for p in plants:
        obj += open[p] * fixedCosts[p]
    for c, t in zip(transport_costs_numpy.flat, transport.flat):
        obj += t * c

    m.set_objective(obj)

    t1 = time.time()
    print(f"POI elapsed time: {t1-t0:.2f} seconds")

    return m


if __name__ == "__main__":
    plants, warehouses, transport_costs, transport_costs_numpy = gen_input(3000, 3000)
    pyoframe_main(plants, warehouses, transport_costs)
    poi_main(plants, warehouses, transport_costs_numpy)
