#date: 2022-06-22T17:25:56Z
#url: https://api.github.com/gists/7ea45b5f6b6ba8e53e8748d3dcdc6a26
#owner: https://api.github.com/users/peytonrunyan

# --- BEFORE ---
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# target_metadata = None

# --- AFTER ---
from models import Base
target_metadata = Base.metadata
# target_metadata = None