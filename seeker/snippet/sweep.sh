#date: 2023-12-08T16:45:59Z
#url: https://api.github.com/gists/ee26b94632ef70c68fa7ca6aa81f7085
#owner: https://api.github.com/users/bjacob

#!/bin/bash

set -eu

amd_events=(
    bp_l2_btb_correct
    bp_dyn_ind_pred
    bp_de_redirect
    ex_ret_brn
    ex_ret_brn_misp
    ex_ret_brn_tkn
    ex_ret_brn_tkn_misp
    ex_ret_brn_far
    ex_ret_near_ret
    ex_ret_near_ret_mispred
    ex_ret_brn_ind_misp
    ex_ret_ind_brch_instr
    ex_ret_cond
    ex_ret_msprd_brnch_instr_dir_msmtch
    ex_ret_uncond_brnch_instr_mispred
    ex_ret_uncond_brnch_instr
    ls_mab_alloc.load_store_allocations
    ls_mab_alloc.hardware_prefetcher_allocations
    ls_mab_alloc.all_allocations
    ls_dmnd_fills_from_sys.local_l2
    ls_dmnd_fills_from_sys.local_ccx
    ls_dmnd_fills_from_sys.near_cache
    ls_dmnd_fills_from_sys.dram_io_near
    ls_dmnd_fills_from_sys.far_cache
    ls_dmnd_fills_from_sys.dram_io_far
    ls_dmnd_fills_from_sys.alternate_memories
    ls_dmnd_fills_from_sys.all
    ls_any_fills_from_sys.local_l2
    ls_any_fills_from_sys.local_ccx
    ls_any_fills_from_sys.local_all
    ls_any_fills_from_sys.near_cache
    ls_any_fills_from_sys.dram_io_near
    ls_any_fills_from_sys.far_cache
    ls_any_fills_from_sys.remote_cache
    ls_any_fills_from_sys.dram_io_far
    ls_any_fills_from_sys.dram_io_all
    ls_any_fills_from_sys.far_all
    ls_any_fills_from_sys.all_dram_io
    ls_any_fills_from_sys.alternate_memories
    ls_any_fills_from_sys.all
    ls_pref_instr_disp.prefetch
    ls_pref_instr_disp.prefetch_w
    ls_pref_instr_disp.prefetch_nta
    ls_pref_instr_disp.all
    ls_inef_sw_pref.data_pipe_sw_pf_dc_hit
    ls_inef_sw_pref.mab_mch_cnt
    ls_inef_sw_pref.all
    ls_sw_pf_dc_fills.local_l2
    ls_sw_pf_dc_fills.local_ccx
    ls_sw_pf_dc_fills.near_cache
    ls_sw_pf_dc_fills.dram_io_near
    ls_sw_pf_dc_fills.far_cache
    ls_sw_pf_dc_fills.dram_io_far
    ls_sw_pf_dc_fills.alternate_memories
    ls_sw_pf_dc_fills.all
    ls_hw_pf_dc_fills.local_l2
    ls_hw_pf_dc_fills.local_ccx
    ls_hw_pf_dc_fills.near_cache
    ls_hw_pf_dc_fills.dram_io_near
    ls_hw_pf_dc_fills.far_cache
    ls_hw_pf_dc_fills.dram_io_far
    ls_hw_pf_dc_fills.alternate_memories
    ls_hw_pf_dc_fills.all
    ls_alloc_mab_count
    l2_request_g1.group2
    l2_request_g1.l2_hw_pf
    l2_request_g1.prefetch_l2_cmd
    l2_request_g1.change_to_x
    l2_request_g1.cacheable_ic_read
    l2_request_g1.ls_rd_blk_c_s
    l2_request_g1.rd_blk_x
    l2_request_g1.rd_blk_l
    l2_request_g1.all_dc
    l2_request_g1.all_no_prefetch
    l2_request_g1.all
    l2_cache_req_stat.ic_fill_miss
    l2_cache_req_stat.ic_fill_hit_s
    l2_cache_req_stat.ic_fill_hit_x
    l2_cache_req_stat.ic_hit_in_l2
    l2_cache_req_stat.ic_access_in_l2
    l2_cache_req_stat.ls_rd_blk_c
    l2_cache_req_stat.ic_dc_miss_in_l2
    l2_cache_req_stat.ls_rd_blk_x
    l2_cache_req_stat.ls_rd_blk_l_hit_s
    l2_cache_req_stat.ls_rd_blk_l_hit_x
    l2_cache_req_stat.ls_rd_blk_cs
    l2_cache_req_stat.dc_hit_in_l2
    l2_cache_req_stat.ic_dc_hit_in_l2
    l2_cache_req_stat.dc_access_in_l2
    l2_cache_req_stat.all
    l2_pf_hit_l2.l2_stream
    l2_pf_hit_l2.l2_next_line
    l2_pf_hit_l2.l2_up_down
    l2_pf_hit_l2.l2_burst
    l2_pf_hit_l2.l2_stride
    l2_pf_hit_l2.l1_stream
    l2_pf_hit_l2.l1_stride
    l2_pf_hit_l2.l1_region
    l2_pf_hit_l2.all
    l2_pf_miss_l2_hit_l3.l2_stream
    l2_pf_miss_l2_hit_l3.l2_next_line
    l2_pf_miss_l2_hit_l3.l2_up_down
    l2_pf_miss_l2_hit_l3.l2_burst
    l2_pf_miss_l2_hit_l3.l2_stride
    l2_pf_miss_l2_hit_l3.l1_stream
    l2_pf_miss_l2_hit_l3.l1_stride
    l2_pf_miss_l2_hit_l3.l1_region
    l2_pf_miss_l2_hit_l3.all
    l2_pf_miss_l2_l3.l2_stream
    l2_pf_miss_l2_l3.l2_next_line
    l2_pf_miss_l2_l3.l2_up_down
    l2_pf_miss_l2_l3.l2_burst
    l2_pf_miss_l2_l3.l2_stride
    l2_pf_miss_l2_l3.l1_stream
    l2_pf_miss_l2_l3.l1_stride
    l2_pf_miss_l2_l3.l1_region
    l2_pf_miss_l2_l3.all
    ic_cache_fill_l2
    ic_cache_fill_sys
    ic_tag_hit_miss.instruction_cache_hit
    ic_tag_hit_miss.instruction_cache_miss
    ic_tag_hit_miss.all_instruction_cache_accesses
    op_cache_hit_miss.op_cache_hit
    op_cache_hit_miss.op_cache_miss
    op_cache_hit_miss.all_op_cache_accesses
    l3_lookup_state.l3_miss
    l3_lookup_state.l3_hit
    l3_lookup_state.all_coherent_accesses_to_l3
    l3_xi_sampled_latency.dram_near
    l3_xi_sampled_latency.dram_far
    l3_xi_sampled_latency.near_cache
    l3_xi_sampled_latency.far_cache
    l3_xi_sampled_latency.ext_near
    l3_xi_sampled_latency.ext_far
    l3_xi_sampled_latency.all
    l3_xi_sampled_latency_requests.dram_near
    l3_xi_sampled_latency_requests.dram_far
    l3_xi_sampled_latency_requests.near_cache
    l3_xi_sampled_latency_requests.far_cache
    l3_xi_sampled_latency_requests.ext_near
    l3_xi_sampled_latency_requests.ext_far
    l3_xi_sampled_latency_requests.all
    ls_locks.bus_lock
    ls_ret_cl_flush
    ls_ret_cpuid
    ls_smi_rx
    ls_int_taken
    ls_not_halted_cyc
    ex_ret_instr
    ex_ret_ops
    ex_div_busy
    ex_div_count
    ex_no_retire.empty
    ex_no_retire.not_complete
    ex_no_retire.other
    ex_no_retire.thread_not_selected
    ex_no_retire.load_not_complete
    ex_no_retire.all
    ls_not_halted_p0_cyc.p0_freq_cyc
    ex_ret_ucode_instr
    ex_ret_ucode_ops
    ex_tagged_ibs_ops.ibs_tagged_ops
    ex_tagged_ibs_ops.ibs_tagged_ops_ret
    ex_ret_fused_instr
    ls_bad_status2.stli_other
    ls_dispatch.ld_dispatch
    ls_dispatch.store_dispatch
    ls_dispatch.ld_st_dispatch
    ls_stlf
    ls_st_commit_cancel2.st_commit_cancel_wcb_full
    ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit
    ls_l1_d_tlb_miss.tlb_reload_coalesced_page_hit
    ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit
    ls_l1_d_tlb_miss.tlb_reload_1g_l2_hit
    ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss
    ls_l1_d_tlb_miss.tlb_reload_coalesced_page_miss
    ls_l1_d_tlb_miss.tlb_reload_2m_l2_miss
    ls_l1_d_tlb_miss.tlb_reload_1g_l2_miss
    ls_l1_d_tlb_miss.all_l2_miss
    ls_l1_d_tlb_miss.all
    ls_misal_loads.ma64
    ls_misal_loads.ma4k
    ls_tlb_flush.all
    bp_l1_tlb_miss_l2_tlb_hit
    bp_l1_tlb_miss_l2_tlb_miss.if4k
    bp_l1_tlb_miss_l2_tlb_miss.if2m
    bp_l1_tlb_miss_l2_tlb_miss.if1g
    bp_l1_tlb_miss_l2_tlb_miss.coalesced_4k
    bp_l1_tlb_miss_l2_tlb_miss.all
    bp_l1_tlb_fetch_hit.if4k
    bp_l1_tlb_fetch_hit.if2m
    bp_l1_tlb_fetch_hit.if1g
    bp_l1_tlb_fetch_hit.all
)

clang++ -Wall -Wextra ~/hook.cc -O2 -shared -fPIC -o /tmp/hook.so

progress=0
seconds_start=$SECONDS
progress_end=${#amd_events[@]}
for event in ${amd_events[@]}; do
  seconds_now=$SECONDS
  if [[ $seconds_now == $seconds_start ]]; then
    eta="..."
  else
    eta="$(((progress_end - progress) * (seconds_now - seconds_start) / progress))"
  fi
  echo "[$progress/$progress_end] [ETA: ${eta}s] event: $event ..."
  progress=$((progress + 1))
  if [[ -z "$event" ]]; then
    continue
  fi
  tmpdir="/tmp/sweep/$event"
  rm -rf "$tmpdir"
  mkdir -p "$tmpdir"
  IREE_HOOK_PERF_EVENT_TYPES="$event" \
    IREE_HOOK_OUTPUT_CSV="$tmpdir" \
    IREE_HOOK_SKIP_START_MS=3000 \
    IREE_HOOK_FILTER_NAME=run_forward_dispatch_104_batch_mmt4d_32x1x344x16x1x32x8_i32 \
    LD_PRELOAD=/tmp/hook.so \
    taskset ff $HOME/iree-build/tools/iree-benchmark-module \
      --module=/tmp/Llama_2_7b_chat_hf.vmfb \
      --function=run_forward \
      --device=local-task \
      --input=1x1xi64 \
      --parameters=model=$HOME/SHARK-Turbine/Llama_2_7b_chat_hf_f32_int4.safetensors \
      --benchmark_repetitions=20 \
      > "$tmpdir/stdout.txt" \
      2> "$tmpdir/stderr.txt"
done
