#date: 2022-08-30T16:46:15Z
#url: https://api.github.com/gists/c215f73591f4cc0ad5b3a9d1e096b66c
#owner: https://api.github.com/users/silee2

#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
dialects=(mhlo builtin acc affine amdgpu amx arith arm_neon arm_sve async \
          bufferization cf complex dlti emitc func gpu linalg llvm math memref \
          ml_program nvgpu nvvm omp pdl pdl_interp quant rocdl scf shape \
          sparse_tensor spv tensor vector x86_vector tosa transform)

tensor=(cast collapse_shape dim expand_shape extract extract_slice \
        from_element generate insert insert_slice pad parallel_insert_slice \
        rank reshape splat yield)

arith=(addf addi addui_carry andi bitcast ceildivsi ceildivui cmpf cmpi \
       constant divf divsi divui extf extsi extui fptosi fptoui floordivsi \
       index_cast maxf maxsi maxui minf minsi minui mulf muli negf ori remf remsi \
       remui sitofp shli shrsi shrui subf subi truncf trunci uitofp xori select)

scf=(condition execute_region for foreach_thread if parallel \
     foreach_thread\\.perform_concurrently reduce reduce\\.return while yield)

linalg=(batch_matmul batch_matvec conv_1d_nwc_wcf conv_1d conv_2d_nchw_fchw \
        conv_2d_ngchw_fgchw conv_2d_nhwc_fhwc conv_2d_nhwc_hwcf conv_2d_nhwc_hwcf_q \
        conv_2d conv_3d_ndhwc_dhwcf conv_3d copy depthwise_conv_1d_nwc_wc \
        depthwise_conv_1d_nwc_wcm depthwise_conv_2d_nchw_chw depthwise_conv_2d_nhwc_hwc \
        depthwise_conv_2d_nhwc_hwc_q depthwise_conv_2d_nhwc_hwcm depthwise_conv_2d_nhwc_hwcm_q \
        depthwise_conv_3d_ndhwc_dhwc depthwise_conv_3d_ndhwc_dhwcm dot elemwise_binary \
        elemwise_unary fill fill_rng_2d generic index init_tensor yield matmul matmul_unsigned \
        matvec mmt4d pooling_nchw_max pooling_nchw_sum pooling_ndhwc_max pooling_ndhwc_min \
        pooling_ndhwc_sum pooling_nhwc_max pooling_nhwc_max_unsigned pooling_nhwc_min \
        pooling_nhwc_min_unsigned pooling_nhwc_sum quantized_batch_matmul quantized_matmul vecmat)

for f in *.mlir
do
  echo "================================================================================="
  echo "  Processing:  $f"
  echo "================================================================================="
  # take action on each file. $f store current file name
  #cat "$f"
  all=$(rg --no-filename --count-matches --include-zero "[[:alnum:]_]+\.[[:alnum:]_]+[[:blank:]|\"]" "$f")
  echo "all dialect usage: $all"

  for i in "${dialects[@]}"
  do
    cnt=$(rg --no-filename --count-matches --include-zero "$i\." "$f")
    if [ "$cnt" -ne "0" ]; then
      echo "$i dialect usage: $cnt";
      eval "for j in \"\${$i[@]}\";\\
            do\\
              icnt=\$(rg --no-filename --count-matches --include-zero \"\$i\.\$j[[:blank:]|\\\"]\" \"\$f\")
              if [ \"\$icnt\" -ne \"0\" ]; then \\
                echo \"  \$i.\$j: \$icnt\";\\
              fi \\
            done"
    fi
  done
done    