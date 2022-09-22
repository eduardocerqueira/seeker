#date: 2022-09-22T17:24:38Z
#url: https://api.github.com/gists/49c58ad32ef1b2f6da0c581a093bfbef
#owner: https://api.github.com/users/gmh5225

from idaapi import *
from idautils import *
from idc import *
from ida_funcs import *

from miasm.analysis.binary import Container
from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB
from miasm.ir.symbexec import SymbolicExecutionEngine
from miasm.core.bin_stream_ida import bin_stream_ida

from miasm.core.asmblock import AsmBlockBad
from miasm.expression.expression import ExprInt

import miasm.arch.x86.regs as REGS

# Stolen from MIASM example
def guess_machine(addr=None):
    "Return an instance of Machine corresponding to the IDA guessed processor"

    processor_name = get_inf_attr(INF_PROCNAME)
    info = idaapi.get_inf_structure()

    if info.is_64bit():
        size = 64
    elif info.is_32bit():
        size = 32
    else:
        size = None

    if processor_name == "metapc":
        size2machine = {
            64: "x86_64",
            32: "x86_32",
            None: "x86_16",
        }

        machine = Machine(size2machine[size])

    elif processor_name == "ARM":
        # TODO ARM/thumb
        # hack for thumb: set armt = True in globals :/
        # set bigendiant = True is bigendian
        # Thumb, size, endian
        info2machine = {(True, 32, True): "armtb",
                        (True, 32, False): "armtl",
                        (False, 32, True): "armb",
                        (False, 32, False): "arml",
                        (False, 64, True): "aarch64b",
                        (False, 64, False): "aarch64l",
                        }

        # Get T reg to detect arm/thumb function
        # Default is arm
        is_armt = False
        if addr is not None:
            t_reg = GetReg(addr, "T")
            is_armt = t_reg == 1

        is_bigendian = info.is_be()
        infos = (is_armt, size, is_bigendian)
        if not infos in info2machine:
            raise NotImplementedError('not fully functional')
        machine = Machine(info2machine[infos])

        from miasm.analysis.disasm_cb import guess_funcs, guess_multi_cb
        from miasm.analysis.disasm_cb import arm_guess_subcall, arm_guess_jump_table
        guess_funcs.append(arm_guess_subcall)
        guess_funcs.append(arm_guess_jump_table)

    elif processor_name == "msp430":
        machine = Machine("msp430")
    elif processor_name == "mipsl":
        machine = Machine("mips32l")
    elif processor_name == "mipsb":
        machine = Machine("mips32b")
    elif processor_name == "PPC":
        machine = Machine("ppc32b")
    else:
        print(repr(processor_name))
        raise NotImplementedError('not fully functional')

    return machine

def get_jcc_map():
    return [
        ('JO',   'and', [('eq', REGS.of, 1)]),
        ('JNO',  'and', [('eq', REGS.of, 0)]),
        ('JB',   'and', [('eq', REGS.cf, 1)]),
        ('JC',   'and', [('eq', REGS.cf, 1)]),
        ('JNAE', 'and', [('eq', REGS.cf, 1)]),
        ('JNB',  'and', [('eq', REGS.cf, 0)]),
        ('JNC',  'and', [('eq', REGS.cf, 0)]),
        ('JAE',  'and', [('eq', REGS.cf, 0)]),
        ('JZ',   'and', [('eq', REGS.zf, 1)]),
        ('JE',   'and', [('eq', REGS.zf, 1)]),
        ('JNZ',  'and', [('eq', REGS.zf, 0)]),
        ('JNE',  'and', [('eq', REGS.zf, 0)]),
        ('JBE',  'or',  [('eq', REGS.cf, 1), ('eq', REGS.zf, 1)]),
        ('JNA',  'or',  [('eq', REGS.cf, 1), ('eq', REGS.zf, 1)]),
        ('JNBE', 'and', [('eq', REGS.cf, 0), ('eq', REGS.zf, 0)]),
        ('JA',   'and', [('eq', REGS.cf, 0), ('eq', REGS.zf, 0)]),
        ('JS',   'and', [('eq', REGS.nf, 1)]),
        ('JNS',  'and', [('eq', REGS.nf, 0)]),
        ('JP',   'and', [('eq', REGS.pf, 1)]),
        ('JPE',  'and', [('eq', REGS.pf, 1)]),
        ('JNP',  'and', [('eq', REGS.pf, 0)]),
        ('JPO',  'and', [('eq', REGS.pf, 0)]),
        ('JL',   'and', [('neq', REGS.nf, REGS.of)]),
        ('JNGE', 'and', [('neq', REGS.nf, REGS.of)]),
        ('JNL',  'and', [('eq', REGS.nf, REGS.of)]),
        ('JGE',  'and', [('eq', REGS.nf, REGS.of)]),
        ('JLE',  'or',  [('eq', REGS.zf, 1), ('neq', REGS.nf, REGS.of)]),
        ('JNG',  'or',  [('eq', REGS.zf, 1), ('neq', REGS.nf, REGS.of)]),
        ('JNLE', 'and', [('eq', REGS.zf, 0), ('eq', REGS.nf, REGS.of)]),
        ('JG',   'and', [('eq', REGS.zf, 0), ('eq', REGS.nf, REGS.of)]),
        ('JCXZ', 'and', [('eq', REGS.CX, 0)]),
        ('JECXZ','and', [('eq', REGS.ECX, 0)]),
        ('JRCXZ','and', [('eq', REGS.RCX, 0)])
    ]

"""
Get Jcc infos
"""
def get_jcc_info(jcc_map, jcc_name):
    info = next((x for x in jcc_map if x[0] == jcc_name), None)
    return info
 
__PLUGIN_NAME__ = 'Opaque Predicates Remover Plugin'

__1BYTE_NOP__ = 0x90

class op_plugin_t( idaapi.plugin_t ):

    flags = PLUGIN_UNL
    comment = ('a plugin to automatically remove constant opaque predicates and junk bytes that breaks dissassembly')
    help = ''
    wanted_name = __PLUGIN_NAME__
    wanted_hotkey = None

    # Object's methods
    def __init__( self ) -> None:
        super( op_plugin_t, self ).__init__()
        self._unreachable_blocks = []
        self._start_ea = None
        self._bs = bin_stream_ida()
        self._locdb = LocationDB()
        self._machine = guess_machine( None )
        self._disasm = self._machine.dis_engine( self._bs, loc_db = self._locdb )
        self._lifter = self._machine.lifter_model_call( self._disasm.loc_db )
        self._symbex = SymbolicExecutionEngine( self._lifter )
        self._cur_ircfg = None
        self._jcc_map = get_jcc_map()

    # plugin_t overrides
    def init( self ):
        print( f'Initialized {__PLUGIN_NAME__}' )
        return PLUGIN_KEEP

    def run( self, arg ):
        func = ida_funcs.get_func( get_screen_ea() )
        self._start_ea = func.start_ea
        print( f'Analyzing from {hex( self._start_ea )}' )

        self._find_unreachable_blocks()
        print( f'Found {len( self._unreachable_blocks )} unreachable blocks' )

        patched = self._patch_unreachable_blocks()
        print( f'Patched {patched} blocks !' )

    def term( self ):
        self._unreachable_blocks = None

    # Plugin's methods
    def _disas_block_at_addr( self, addr ):
        return self._disasm.dis_block( addr )

    def _disas_multiblock_at_addr( self, addr ):
        return self._disasm.dis_multiblock( addr )

    def _create_ir_from_block( self, block ):
        self._cur_ircfg = self._lifter.new_ircfg()
        self._lifter.add_asmblock_to_ircfg( block, self._cur_ircfg )

    def _eval_register_expr( self, addr, reg ):
        self._symbex.run_block_at( self._cur_ircfg, addr, reg )
        return self._symbex.eval_exprid( reg )

    def _check_opaque_predicate( self, block, info ):
        self._create_ir_from_block( block )
        blk_addr = block.loc_db.get_location_offset( block.loc_key )
        cond = info[ 1 ]
        regs = info[ 2 ]

        exprs = []
        for i, ( c, r, v ) in enumerate( regs ):
            expr = self._eval_register_expr( blk_addr, r )
            if type( expr ) == ExprInt:
                value = expr._arg
                if ( c == 'eq' and value == v ) or ( c == 'neq' and value != v ):
                    exprs.append( True )
                else:
                    exprs.append( False )

        if len( exprs ) == 0:
            return False

        is_opaque = False

        if cond == 'and':
            is_opaque = all( exprs )
        elif cond == 'or':
            is_opaque = any( exprs )

        return is_opaque

    def _find_unreachable_blocks( self ):
        jccs = [ name for name, op, c in self._jcc_map ]
        cfg = self._disas_multiblock_at_addr( self._start_ea )
        blocks = list( cfg.blocks )

        for index, block in enumerate( blocks ):
            if type( block ) is AsmBlockBad:
                continue

            last_insn = block.lines[ -1 ]
            if last_insn.name in jccs:
                jcc_info = get_jcc_info( self._jcc_map, last_insn.name )
                if self._check_opaque_predicate( block, jcc_info ):
                    addr = block.loc_db.get_location_offset( block.loc_key )

                    jcc_addr = last_insn.offset
                    jcc_len = last_insn.l
                    jcc_dst = last_insn.args[ 0 ].loc_key

                    dst_addr = block.loc_db.get_location_offset( jcc_dst )

                    unr_blk_addr = jcc_addr + jcc_len
                    unr_blk_len = dst_addr - unr_blk_addr

                    self._unreachable_blocks.append( ( unr_blk_addr, unr_blk_addr + unr_blk_len ) )

    def _patch_unreachable_blocks( self ):
        patched = 0

        for index, ( block_start, block_end ) in enumerate( self._unreachable_blocks ):
            block_len  = block_end - block_start

            if block_len > 0xff: # If the displacement is > 0xff ( rel16 or rel32 ) it's probably wrong and need to be resolved manually (just nop the junk)
                print( f'Block {index} (start={hex( block_start )}, end={hex( block_end )}, len={block_len}) needs to be reviewed manually' )
            else:
                print( f'Patching block {index} (start={hex( block_start )}, end={hex( block_end )}, len={block_len})' )
                for ea in range ( block_start, block_end ):
                    idaapi.patch_byte( ea, __1BYTE_NOP__ )

                patched += 1

        return patched


def PLUGIN_ENTRY():
    return op_plugin_t()

if __name__ == '__main__':
    plugin = op_plugin_t()
    plugin.init()
    plugin.run( None )
    plugin.term()