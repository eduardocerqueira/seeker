#date: 2024-03-28T17:07:02Z
#url: https://api.github.com/gists/aced97a639d68f8fda0e31b7578b5db3
#owner: https://api.github.com/users/johnnymn

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, Timer
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.3
RANDOM_STATE = 100


async def clock_gen(signal):
    while True:
        signal.value = not signal.value
        await Timer(1, units="ns")


async def reset_dut(rst, duration_ns):
    rst.value = 1
    await Timer(duration_ns, units='ns')
    rst.value = 0


@cocotb.test()
async def test_table(dut):
    cancer = datasets.load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        cancer.data,
        cancer.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = svm.SVC(kernel="linear").fit(x_train, y_train)  # noqa: E999
    y_pred = model.predict(x_test)  # noqa

    sklearn_accuracy = metrics.accuracy_score(y_test, y_pred)
    sklearn_precision = metrics.precision_score(y_test, y_pred)
    sklear_recall = metrics.recall_score(y_test, y_pred)

    print("sklearn model accuracy: ", sklearn_accuracy)
    print("sklearn model precision:", sklearn_precision)
    print("sklearn model recall:", sklear_recall)

    weights = np.round(model.coef_[0], decimals=8)  # noqa
    intercept = np.round(model.intercept_[0], decimals=8)  # noqa

    cocotb.start_soon(clock_gen(dut.clk))  # noqa
    n_features = len(weights)
    y_pred_hdl = []

    for x in x_test:
        dut.b.value = intercept

        for i in range(n_features):
            dut.w[i].value = weights[i]
            dut.x[i].value = np.round(x[i], decimals=8)

        await reset_dut(dut.rst, 1)
        for _ in range(2):
            await RisingEdge(dut.clk)

        y_pred_hdl.append(int(dut.y.value))

    y_pred_hdl = np.array(y_pred_hdl)
    hdl_accuracy = metrics.accuracy_score(y_test, y_pred_hdl)
    hdl_precision = metrics.precision_score(y_test, y_pred_hdl)
    hdl_recall = metrics.recall_score(y_test, y_pred_hdl)

    print("hdl inference accuracy: ", hdl_accuracy)
    print("hdl inference precision:", hdl_precision)
    print("hdl inference recall:", hdl_recall)

    assert sklearn_accuracy == hdl_accuracy
    assert sklearn_precision == hdl_precision
    assert sklear_recall == hdl_recall
