#date: 2021-11-08T17:14:03Z
#url: https://api.github.com/gists/372b3a280ee782efb75b2650d1326976
#owner: https://api.github.com/users/Lyken17

import numpy as np

import tvm
from tvm import relay
from tvm import relay, auto_scheduler
from tvm.relay import testing


SEMVER = '#[version = "0.0.5"]\n'

def assert_graph_equal(lhs, rhs):
    tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars=True)

def roundtrip(expr):
    x = tvm.parser.fromtext(expr.astext())
    assert_graph_equal(x, expr)

# Testing Utilities for full modules.
def parse_module(code):
    mod = tvm.parser.parse(SEMVER + code)
    roundtrip(mod)
    return mod


program = """
def @main(%input0: Tensor[(1, 32, 224, 224), float32], 
        %v0_0_weight: Tensor[(32, 1, 3, 3), float32]) -> Tensor[(1, 32, 224, 224), float32] {
  let %x = %input0;
  %0 = nn.conv2d_transpose(%input0, %v0_0_weight, strides=[1, 1], output_padding=[0, 0],padding=[1, 1, 1, 1], groups=32, channels=32, kernel_size=[3, 3]);
  %0
}
"""


mod = parse_module(program)
print(mod)
print("build [fwd] IR successful")

target = "cuda"
lib = relay.build(mod, target=target, params=None)
print("build [fwd] pass successful")

"""
(base) ➜  tvm-play python tvm_report_mbv2.py
def @main(%input0: Tensor[(1, 32, 224, 224), float32], %v0_0_weight: Tensor[(32, 1, 3, 3), float32]) -> Tensor[(1, 32, 224, 224), float32] {
  let %x: Tensor[(1, 32, 224, 224), float32] = %input0;
  nn.conv2d_transpose(%input0, %v0_0_weight, channels=32, kernel_size=[3, 3], padding=[1, 1, 1, 1], groups=32) /* from_string */ /* ty=Tensor[(1, 32, 224, 224), float32] */
}

build [fwd] IR successful
Register sucessful
Traceback (most recent call last):
  File "tvm_report_mbv2.py", line 81, in <module>
    lib = relay.build(mod, target=target, params=None)
  File "/home/ligeng/Workspace/tvm/python/tvm/relay/build_module.py", line 357, in build
    executor_config, runtime_mod, params = bld_mod.build(
  File "/home/ligeng/Workspace/tvm/python/tvm/relay/build_module.py", line 172, in build
    self._build(mod, target, target_host, executor, mod_name)
  File "/home/ligeng/Workspace/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
tvm._ffi.base.TVMError: Traceback (most recent call last):
  36: TVMFuncCall
  35: _ZNSt17_Function_handlerIFvN3tvm
  34: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
  33: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::NDArray, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, tvm::runtime::NDArray> > > const&, tvm::runtime::String)
  32: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
  31: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::relay::Function, tvm::runtime::String)
  30: tvm::transform::Pass::operator()(tvm::IRModule) const
  29: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  28: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  27: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  26: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  25: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(std::unordered_map<DLDeviceType, tvm::Target, tvm::relay::backend::EnumClassHash, std::equal_to<DLDeviceType>, std::allocator<std::pair<DLDeviceType const, tvm::Target> > >, tvm::runtime::String const&, std::function<void (tvm::relay::Function)>)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(std::unordered_map<DLDeviceType, tvm::Target, tvm::relay::backend::EnumClassHash, std::equal_to<DLDeviceType>, std::allocator<std::pair<DLDeviceType const, tvm::Target> > >, tvm::runtime::String const&, std::function<void (tvm::relay::Function)>)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
  24: tvm::relay::tec::LowerTE(tvm::IRModule const&, std::unordered_map<DLDeviceType, tvm::Target, tvm::relay::backend::EnumClassHash, std::equal_to<DLDeviceType>, std::allocator<std::pair<DLDeviceType const, tvm::Target> > >, tvm::runtime::String const&, std::function<void (tvm::relay::Function)>)
  23: tvm::transform::Pass::operator()(tvm::IRModule) const
  22: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  21: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  20: tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(std::unordered_map<DLDeviceType, tvm::Target, tvm::relay::backend::EnumClassHash, std::equal_to<DLDeviceType>, std::allocator<std::pair<DLDeviceType const, tvm::Target> > >, tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::relay::Function)>)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(std::unordered_map<DLDeviceType, tvm::Target, tvm::relay::backend::EnumClassHash, std::equal_to<DLDeviceType>, std::allocator<std::pair<DLDeviceType const, tvm::Target> > >, tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::relay::Function)>)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
  19: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
  18: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayExprERKS2_EE10InitVTableEvENUlRKNS_7r
  17: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
  16: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
  15: _ZN3tvm5relay9transform22DeviceAwareExprMutator21DeviceAwareVisit
  14: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
  13: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
  12: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayExprERKS2_EE10InitVTableEvENUlRKNS_7r
  11: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::LetNode const*)
  10: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
  9: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayExprERKS2_EE10InitVTableEvENUlRKNS_7r
  8: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
  7: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
  6: tvm::relay::tec::LowerTensorExprMutator::LowerFunction(tvm::relay::Function, tvm::Target)
  5: tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String)
  4: tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>)
  3: tvm::relay::tec::PrimFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
  2: tvm::relay::tec::ScheduleBuilder::Create(tvm::relay::Function const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
  1: tvm::relay::OpImplementation::Schedule(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Target const&)
  0: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&) [clone .cold]
  File "/home/ligeng/Workspace/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 81, in cfun
    rv = local_pyfunc(*pyargs)
  File "/home/ligeng/Workspace/tvm/python/tvm/relay/op/strategy/generic.py", line 51, in wrapper
    return topi_schedule(outs)
  File "/home/ligeng/Workspace/tvm/python/tvm/autotvm/task/topi_integration.py", line 236, in wrapper
    raise RuntimeError("Cannot find workload in attribute of this schedule")
RuntimeError: Cannot find workload in attribute of this schedule
"""

# relay.op.strategy.cuda
"""
@conv2d_transpose_strategy.register(["cuda", "gpu"])
def conv2d_transpose_strategy_cuda(attrs, inputs, out_type, target):
    """conv2d_transpose cuda strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    # assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    if groups == 1:
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.cuda.conv2d_transpose_nchw),
            wrap_topi_schedule(topi.cuda.schedule_conv2d_transpose_nchw),
            name="conv2d_transpose_nchw.cuda",
        )
    else:
        # raise NotImplementedError("CUDA schedule is not enable for conv2d transpose when groups > 1. See https://github.com/apache/tvm/pull/9465 for details.")
        # FIXME: Here should be a specialized implementation and schedule instead general one.
        strategy.add_implementation(
            wrap_compute_group_conv2d_transpose(topi.nn.group_conv2d_transpose_nchw),
            wrap_topi_schedule(topi.cuda.schedule_group_conv2d_transpose_nchw),
            name="group_conv2d_transpose_nchw.cuda",
        )
        print("Register sucessful")
    return strategy
"""

# topi.cuda.conv2d_transposed_nchw
"""
@autotvm.register_topi_schedule("group_conv2d_transpose_nchw.cuda")
def schedule_group_conv2d_transpose_nchw(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    
    def _fallback_schedule(N, F, Y, X):
        # pylint: disable=unused-argument
        # split N (batch dimension)
        if N > 1:
            cfg["tile_n"] = SplitEntity([-1, 1, 1, 4])
        else:
            cfg["tile_n"] = SplitEntity([1, 1, 1, 1])
        # split F (output channel dimension)
        if F > 1:
            cfg["tile_f"] = SplitEntity([-1, 1, 64, 1])
        # split Y (height dimension)
        y_split_factor = 1
        for candidate in range(5, 17):
            if Y % candidate == 0:
                y_split_factor = candidate
                break
        cfg["tile_y"] = SplitEntity([-1, 1, 1, y_split_factor])
        # split X (width dimension)
        x_split_factor = 1
        for candidate in range(5, 17):
            if X % candidate == 0:
                x_split_factor = candidate
                break
        cfg["tile_x"] = SplitEntity([-1, x_split_factor, 1, 1])
        # split RC (input channel dimension, which is a reduction axis)
        cfg["tile_rc"] = SplitEntity([-1, 1, 16])
        # other configurations
        cfg["fuse_yx"] = OtherOptionEntity(False)
        cfg["unroll_explicit"] = OtherOptionEntity(True)
        cfg["auto_unroll_max_step"] = OtherOptionEntity(1500)

    def _callback(op):
        if op.tag == "group_conv2d_transpose_nchw":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, y, x = s[conv].op.axis
            rc = s[conv].op.reduce_axis[0]
            # TODO(@kevinthesun): Support tuning/optimization for dynamic shape.
            bs = pad_data.shape[0]
            n_tuning_axis = n if isinstance(bs, tvm.tir.IntImm) else 1
            cfg.define_split("tile_n", cfg.axis(n_tuning_axis), num_outputs=4)
            cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
            cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
            cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
            cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
            cfg.define_knob("auto_unroll_max_step", [64, 512, 1500])

            target = tvm.target.Target.current()
            if target.kind.name in ["nvptx", "rocm"]:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            if cfg.is_fallback:
                N, F, Y, X = get_const_tuple(conv.shape)
                if not isinstance(N, int):
                    N = 1
                _fallback_schedule(N, F, Y, X)

            ##### space definition end #####

            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, "local")
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope("local")
                OL = conv

            # create cache stage
            s[pad_data].set_scope("shared")
            AA = pad_data
            WW = s.cache_read(kernel, "shared", [OL])

            # tile and bind spatial axes
            n, f, y, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            s[output].reorder(bn, bf, by, bx, vn, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
            s[output].bind(bn, te.thread_axis("blockIdx.z"))
            s[output].bind(bf, te.thread_axis("blockIdx.y"))
            s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
            s[output].bind(vn, te.thread_axis("vthread"))
            s[output].bind(vf, te.thread_axis("vthread"))
            s[output].bind(vy, te.thread_axis("vthread"))
            s[output].bind(vx, te.thread_axis("vthread"))

            cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf

            if cfg["fuse_yx"].val:
                s[output].bind(tn, te.thread_axis("threadIdx.z"))
                s[output].bind(tf, te.thread_axis("threadIdx.y"))
                tyx = s[output].fuse(ty, tx)
                s[output].bind(s[output].fuse(ty, tx), te.thread_axis("threadIdx.x"))
                s[OL].compute_at(s[output], tyx)

                # number of threads
                n_tz = cfg["tile_n"].size[2]
                n_ty = cfg["tile_f"].size[2]
                n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
            else:
                s[output].bind(s[output].fuse(tn, tf), te.thread_axis("threadIdx.z"))
                s[output].bind(ty, te.thread_axis("threadIdx.y"))
                s[output].bind(tx, te.thread_axis("threadIdx.x"))
                s[OL].compute_at(s[output], tx)

                # number of threads
                n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
                n_ty = cfg["tile_y"].size[2]
                n_tx = cfg["tile_x"].size[2]

            # tile reduction axes
            n, f, y, x = s[OL].op.axis
            rc, ry, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, ry, rx, rci, n, f, y, x)

            s[AA].compute_at(s[OL], rx)
            s[WW].compute_at(s[OL], rx)

            # cooperative fetching
            for load in [AA, WW]:
                n, f, y, x = s[load].op.axis
                fused = s[load].fuse(f, y, x)
                tz, fused = s[load].split(fused, nparts=n_tz)
                ty, fused = s[load].split(fused, nparts=n_ty)
                tx, fused = s[load].split(fused, nparts=n_tx)
                s[load].bind(tz, te.thread_axis("threadIdx.z"))
                s[load].bind(ty, te.thread_axis("threadIdx.y"))
                s[load].bind(tx, te.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
            s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    traverse_inline(s, outs[0].op, _callback)

    return s
"""


