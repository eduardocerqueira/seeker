#date: 2021-12-20T17:01:02Z
#url: https://api.github.com/gists/a704e6cc5c9a84d3936725f446017864
#owner: https://api.github.com/users/dormeir999

# Helper functions

def models_table(files_in_dir=None, models_dir=configs['MODELS_DIR'], use_models_dir=True):
    if use_models_dir is False:
        models_dir = os.getcwd()
    if files_in_dir is None:
        files_in_dir = os.listdir(models_dir)
        if ".gitignore" in files_in_dir:
            files_in_dir.remove(".gitignore")
    file_modified_time = [
datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(str(models_dir), str(file)))).isoformat() for file
        in files_in_dir]
    file_sizes = [
        format(os.path.getsize(os.path.join(models_dir, file)) / (1024 * 1024), f".{configs['N_DECIMAL_POINTS']}f") for
        file in files_in_dir]
    files_with_time = pd.DataFrame(data=[files_in_dir, file_modified_time, file_sizes],
                                   index=['Model', 'Last modified', 'Size in MB']).T
    return files_with_time
  
# Implementation

table = st.empty()
col3, col4, col5 = st.columns((1, 1, 15))
files_in_dir = os.listdir(models_abs_dir)
table.write(models_table(files_in_dir))
if col3.button("Save"):
    try:
        with open(str(state.model_saving_name), 'wb') as outf:
            pickle.dump(state._state['data'], outf)  # , protocol=4)
            st.success("Saved " + str(state.model_saving_basename))
    except:
        st.write("Could not save " + str(state.model_saving_basename) + " :")
        e = sys.exc_info()
        st.error(e)
    finally:
        table.write(models_table())
if col3.button("Load"):
    try:
        with open(str(state.model_saving_name), 'rb') as inf:
            state._state['data'] = pickle.load(inf)
            state.message = "Loaded " + str(state.model_saving_basename)
    except:
        col3.info("Could not load " + str(state.model_saving_basename))
if col4.button("Delete"):
    try:
        os.remove(state.model_saving_name)
        col3.success("Deleted " + str(state.model_saving_basename))
    except:
        col3.write("Could not delete " + str(state.model_saving_name) + " :")
        state.error = sys.exc_info()
    finally:
        table.write(models_table())