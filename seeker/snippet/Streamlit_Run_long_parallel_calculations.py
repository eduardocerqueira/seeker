#date: 2021-12-20T17:16:13Z
#url: https://api.github.com/gists/6dc80045c8da44245b40e9b0f446837b
#owner: https://api.github.com/users/dormeir999

# Helper functions

def get_proc_name(proc):
    try:
        if "jupyter lab --allow-root" in proc.cmdline():
            return "coding_bench"
        return os.path.basename([line for line in proc.cmdline() if ".py" in line][0]).split(".py")[0]
    except IndexError:
        return ""
    except psutil.NoSuchProcess:
        return ""
      
def cpu_time_to_duration(cpu_time):
    return time.strftime("%H:%M:%S", time.localtime(cpu_time))
  
def cpu_time_to_calender(cpu_time):
   time.localtime(cpu_time))
    
def create_update_process_df(state):
    state.process_df = pd.DataFrame(columns=['name', 'elapsed H:M:S', 'started', 'pid'])
    if configs['SHOW_CMDLINE']:
        state.process_df = pd.DataFrame(columns=['name', 'elapsed H:M:S', 'started', 'pid', 'cmd-line'])
    procs = psutil.Process(os.getpid()).children()
    for proc in procs:
        try:
            name = get_proc_name(proc)
            if not name == "free_coding":
                elapsed = cpu_time_to_duration(time.time() - proc.create_time())
                started = cpu_time_to_calender(proc.create_time())
                pid = proc.pid
                if configs['SHOW_CMDLINE']:
                    cmdline = proc.cmdline()
                if configs['SHOW_CMDLINE']:
                    state.process_df = state.process_df.append(
                        {'name': name, 'elapsed H:M:S': elapsed, 'started': started, 'pid': pid,
                         'cmdline': cmdline},
                        ignore_index=True)
                else:
                    state.process_df = state.process_df.append(
                        {'name': name, 'elapsed H:M:S': elapsed, 'started': started, 'pid': pid},
                        ignore_index=True)
        except:
            pass
    return procs
  
def process_running_without_jupyter():
    try:
        return [proc.name() for proc in psutil.Process(os.getpid()).children() if "jupyter lab --allow-root" not in proc.cmdline()]
    except psutil.NoSuchProcess:
        return []
      
def updated_processes_table(process_table_sidebar, state):
    create_update_process_df(state)
    process_table_sidebar.dataframe(state.process_df)
    st.write("")
    
def stop_proc_and_child_procs(process, key=""):
    stop_model_button = st.empty()
    stop_message = st.empty()
    try:
        children = process.children(recursive=True)
        while len(children) == 0:
            children = process.children(recursive=True)
        for process in children:
            process.send_signal(sig=signal.SIGTERM)
        children = process.children(recursive=True)
        if len(children) == 0:
            pass
    except:
        pass
      
# Implementation

st.sidebar.header(":running: Processes table")
process_table_sidebar = st.sidebar.empty()
col7, col8 = st.sidebar.beta_columns(2)
procs = create_update_process_df(state)
stop_processes_button = col7.button("Stop ALL")
with col7:
    if stop_processes_button:
        while process_running_without_jupyter():
            procs = psutil.Process(os.getpid()).children()
            for proc in procs:
                if get_proc_name(proc) != "free_coding":
                    stop_proc_and_child_procs(proc)
    if procs:
        for proc in procs:
            if get_proc_name(proc) != "free_coding":
                try:
                    if st.button(f"Stop {get_proc_name(proc)}"):
                        stop_proc_and_child_procs(proc)
                except:
                    if st.button(f"Stop {get_proc_name(proc)}_{proc.pid}"):
                        stop_proc_and_child_procs(proc)

with col8:
    if st.button("Refresh", key="processes"):
        create_update_process_df(state)

process_table_sidebar.dataframe(state.process_df)