#date: 2025-01-16T16:57:14Z
#url: https://api.github.com/gists/98659e811e90a8767789a60208edf02d
#owner: https://api.github.com/users/AverageBusinessUser

def get_xref_hexrays(target_data_ea):
    """
    gets a list of the hexrays that uses the provided address
    target_data_ea - target address
    returns a list dicts where the target is used
    [{'function': 'boot_thing', 'code': 'if ( boot_thing(2u, &blah) )'},]
    
    """
    results = []
    query = ''
    target_data_name = ida_name.get_name(target_data_ea)
    # Ensure the decompiler is available
    if not ida_hexrays.init_hexrays_plugin():
        logging.error(f'Hex-Rays decompiler is not available.')
        return results
    
    target_xrefs = []
    xrefs = idautils.XrefsTo(target_data_ea)
    for xref in xrefs:
        #get a reference to the function
        curfunc = ida_funcs.get_func_name(xref.frm)
        curfunc_t = ida_funcs.get_func(xref.frm)
        if curfunc:
            target_xrefs.append(curfunc_t.start_ea)

    # Iterate through all functions in the binary
    for ea in set(target_xrefs):
        func_name = ida_funcs.get_func_name(ea)
        
        try:
            # Decompile the function
            cfunc = ida_hexrays.decompile(ea)
            if not cfunc:
                logging.error(f'failed to decompile function at {hex(ea)}')
                continue

            # Get the decompiled code as text
            decompiled_text = cfunc.get_pseudocode()

            # Search for the target function name in each line
            for line_number, line in enumerate(decompiled_text, 1):
                # Remove tags to get clean text
                line_text = ida_lines.tag_remove(line.line)
                
                if target_data_name in line_text:
                    results.append((func_name, line_number, line_text.strip()))
        
        except ida_hexrays.DecompilationFailure as e:
            logging.error(f'decompilation failed for function at {hex(ea)}: {str(e)}')
    res = []
    if results:
        print(f'{target_data_name} is referenced in the following locations:\n')
        # Build the query

        if results:
            for ref in results:
                print(f'{ref[0]}: {ref[2]}')
                res.append({'function':ref[0],'code':ref[2]})
        
    return res