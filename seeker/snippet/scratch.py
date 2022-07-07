#date: 2022-07-07T17:03:17Z
#url: https://api.github.com/gists/ca9eb6b8ee0799b16d424f8a9e99e35b
#owner: https://api.github.com/users/Strilanc

from typing import Optional, Dict

import cirq


def compose(*gates: cirq.Gate) -> cirq.Gate:
    matrix = cirq.unitary(gates[0])
    for g in gates[1:]:
        matrix = cirq.unitary(g) @ matrix
    return cirq.MatrixGate(matrix)


def init_and3(c2: cirq.Qid, c3: cirq.Qid, c4: cirq.Qid, *, t: cirq.Qid) -> cirq.OP_TREE:
    yield compose(cirq.H, cirq.T).on(t)
    yield cirq.CX(c2, t)
    yield compose(cirq.T**-1, cirq.H, cirq.T**-1).on(t)
    yield cirq.CX(c4, t)
    yield cirq.T(t), cirq.CX(c3, c4)
    yield cirq.CX(c3, t), cirq.T(c4)
    yield cirq.T(t)**-1, cirq.CX(c3, c4)
    yield cirq.CX(c4, t)
    yield cirq.T(t), cirq.T(c3)**-1, cirq.T(c4)**-1
    yield cirq.CX(c3, t)
    yield compose(cirq.H, cirq.T).on(t)
    yield cirq.CX(c2, t)
    yield compose(cirq.T**-1, cirq.H, cirq.S**-1).on(t)


def init_and4(c1: cirq.Qid, c2: cirq.Qid, c3: cirq.Qid, c4: cirq.Qid, *, t: cirq.Qid) -> cirq.OP_TREE:
    yield compose(cirq.H, cirq.T**0.5).on(t)
    yield cirq.CX(c2, t)
    yield cirq.T(t)**-0.5
    yield cirq.CX(c1, t)
    yield cirq.T(t)**0.5
    yield cirq.CX(c2, t)
    yield compose(cirq.T**-0.5, cirq.H, cirq.T**-1).on(t)
    yield cirq.CX(c4, t)
    yield cirq.T(t), cirq.CX(c3, c4)
    yield cirq.CX(c3, t), cirq.T(c4)
    yield cirq.T(t)**-1, cirq.CX(c3, c4)
    yield cirq.CX(c4, t)
    yield cirq.T(t), cirq.T(c3)**-1, cirq.T(c4)**-1
    yield cirq.CX(c3, t)
    yield compose(cirq.H, cirq.T**0.5).on(t)
    yield cirq.CX(c2, t)
    yield cirq.T(t)**-0.5
    yield cirq.CX(c1, t)
    yield cirq.T(t)**0.5
    yield cirq.CX(c2, t)
    yield compose(cirq.T**-0.5, cirq.H, cirq.S**-1).on(t)


def negate_and4_a1(c1: cirq.Qid, c2: cirq.Qid, c3: cirq.Qid, c4: cirq.Qid, *, anc: cirq.Qid) -> cirq.OP_TREE:
    yield compose(cirq.H, cirq.T).on(anc)
    yield cirq.CX(c1, anc)
    yield cirq.T(anc)**-1
    yield cirq.CX(c2, anc)
    yield cirq.T(anc)
    yield cirq.CX(c1, anc)
    yield compose(cirq.T**-1, cirq.H, cirq.T).on(anc)
    yield cirq.CX(c3, anc)
    yield cirq.T(anc)**-1, cirq.CX(c4, c3)
    yield cirq.CX(c4, anc), cirq.T(c3) ** -1
    yield cirq.T(anc), cirq.CX(c4, c3)
    yield cirq.CX(c3, anc)
    yield cirq.T(anc) ** -1, cirq.T(c4), cirq.T(c3)
    yield cirq.CX(c4, anc)
    yield compose(cirq.H, cirq.T).on(anc)
    yield cirq.CX(c1, anc)
    yield cirq.T(anc) ** -1
    yield cirq.CX(c2, anc)
    yield cirq.T(anc)
    yield cirq.CX(c1, anc)
    yield compose(cirq.T**-1, cirq.H).on(anc)


reset = cirq.ResetChannel()

def mb_cccz(c1: cirq.Qid, c2: cirq.Qid, c3: cirq.Qid, c4: cirq.Qid, *, anc: cirq.Qid, measure_key: str, condition_key: Optional[str] = None) -> cirq.OP_TREE:
    cs = [] if condition_key is None else [condition_key]
    yield compose(cirq.H, cirq.T).on(anc)
    yield cirq.CX(c1, anc)
    yield cirq.T(anc)**-1
    yield cirq.CX(c2, anc)
    yield cirq.T(anc)
    yield cirq.CX(c1, anc)
    yield cirq.CX(c3, anc)
    yield cirq.T(anc)**-1
    yield cirq.CX(c4, anc).with_classical_controls(*cs)
    yield cirq.T(anc)
    yield cirq.CX(c3, anc)
    yield compose(cirq.T**-1, cirq.X**-0.5).on(anc), cirq.H(c1), cirq.H(c3)
    yield cirq.CX(c4, anc).with_classical_controls(*cs)
    yield cirq.measure(anc, key=measure_key), cirq.CX(c4, c3).with_classical_controls(*cs)
    yield cirq.Moment(
        reset(anc),
        cirq.CX(c2, c1).with_classical_controls(measure_key),
        cirq.CX(c4, c3).with_classical_controls(measure_key, *cs),
    )
    yield cirq.H(c1), cirq.H(c3)


def make_c14x_circuit_using_measurement() -> cirq.Circuit:
    qs = cirq.LineQubit.range(20)
    inputs = qs[:15]
    a, b, c, d, e = qs[15:]

    group_into_ancillae = cirq.Circuit(
        init_and4(*qs[0:4], t=b),
        init_and4(*qs[4:8], t=c),
        init_and4(*qs[8:12], t=d),
        init_and3(*qs[12:15], t=e),
    )
    return cirq.Circuit.concat_ragged(
        cirq.Circuit(cirq.H(inputs[-1])),
        group_into_ancillae,
        cirq.Circuit(mb_cccz(e, d, c, b, anc=a, measure_key="bcde")),
        cirq.Circuit(
            cirq.Moment(
                cirq.H.on_each(b, c, d, e),
            ),
            cirq.Moment(
                [cirq.I(q) for q in qs[:16]],
                cirq.measure(b, key="0123"),
                cirq.measure(c, key="4567"),
                cirq.measure(d, key="8-9-10-11"),
                cirq.measure(e, key="12-13-14"),
            ),
            cirq.Moment(
                reset.on_each(b, c, d, e),
            ),
        ),
        cirq.Circuit(mb_cccz(*qs[0:4], anc=b, measure_key="b", condition_key="0123")),
        cirq.Circuit(mb_cccz(*qs[4:8], anc=c, measure_key="c", condition_key="4567")),
        cirq.Circuit(mb_cccz(*qs[8:12], anc=d, measure_key="d", condition_key="8-9-10-11")),
        cirq.Circuit(
            cirq.decompose_once(cirq.CCZ(*qs[12:15]).with_classical_controls(cirq.MeasurementKey("12-13-14"))),
        ),
        cirq.Circuit(cirq.H(inputs[-1])),
    )


def make_c14x_circuit_without_measurement() -> cirq.Circuit:
    qs = cirq.LineQubit.range(20)
    inputs = qs[:15]
    a, b, c, d, e = qs[15:]
    group_into_ancillae = cirq.Circuit(
        init_and4(*qs[0:4], t=b),
        init_and4(*qs[4:8], t=c),
        init_and4(*qs[8:12], t=d),
        init_and3(*qs[12:15], t=e),
    )
    return cirq.Circuit.concat_ragged(
        cirq.Circuit(cirq.H(inputs[-1])),
        group_into_ancillae,
        cirq.Circuit(negate_and4_a1(e, d, c, b, anc=a)),
        cirq.inverse(group_into_ancillae),
        cirq.Circuit(cirq.H(inputs[-1])),
    )


def make_test_circuit(impl: cirq.Circuit) -> cirq.Circuit:
    qs = cirq.LineQubit.range(20)
    inputs = qs[:15]
    a, b, c, d, e = qs[15:]
    return cirq.Circuit(
        [cirq.H(q) for q in inputs[:-1]],
        impl,
        cirq.X(inputs[-1]).controlled_by(*inputs[:-1]),
        [cirq.H(q) for q in inputs[:-1]],
        reset(a),
        reset(b),
        reset(c),
        reset(d),
        reset(e),
    )


def fold_single_qubit_operations(c: cirq.Circuit) -> cirq.Circuit:
    def sq(m: cirq.Moment) -> Dict[cirq.Qid, cirq.Operation]:
        return {op.qubits[0]: op for op in m if len(op.qubits) == 1 and cirq.has_unitary(op)}

    moments = []
    prev_m = c[0]
    prev = sq(prev_m)
    for k in range(1, len(c)):
        cur_m = c[k]
        cur = sq(cur_m)
        common = cur.keys() & prev.keys()
        cur_m = cur_m.without_operations_touching(common)
        prev_m = prev_m.without_operations_touching(common).with_operations(
            compose(prev[q].gate, cur[q].gate).on(q)
            for q in common
        )
        if cur_m:
            moments.append(prev_m)
            prev = cur
            prev_m = cur_m
    if prev_m:
        moments.append(prev_m)
    return cirq.Circuit(cirq.Circuit(moments).all_operations())


def main():
    impl1 = make_c14x_circuit_without_measurement()
    impl2 = make_c14x_circuit_using_measurement()
    impl1 = fold_single_qubit_operations(impl1)
    impl2 = fold_single_qubit_operations(impl2)
    for op in impl1.all_operations():
        assert op.gate == cirq.CX or (len(op.qubits) == 1 and cirq.has_unitary(op))
    for op in impl2.all_operations():
        if isinstance(op, cirq.ClassicallyControlledOperation):
            op = op._sub_operation
        assert op.gate == cirq.CX or (len(op.qubits) == 1 and cirq.has_unitary(op)) or op.gate == reset or type(op.gate) == cirq.MeasurementGate
    print(impl2.to_text_diagram(transpose=True))
    print(repr(impl2))
    print()
    print("depth without measurement", len(impl1))
    print("depth with measurement+feedback", len(impl2))
    test_circuits = [make_test_circuit(impl1), make_test_circuit(impl2)]
    print("testing...")
    for k in range(50):
        print("attempt", k, "distance:", end=' ')
        for test_circuit in test_circuits:
            test_state = cirq.Simulator().simulate(test_circuit).final_state_vector
            p = test_state[0] * test_state[0].conj()
            err = abs(p - 1)
            print(err, end=' ')
            assert err < 1e-5
        print()
    print("PASS")


if __name__ == '__main__':
    main()
