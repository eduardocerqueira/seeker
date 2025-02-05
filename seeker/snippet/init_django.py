#date: 2025-02-05T17:06:57Z
#url: https://api.github.com/gists/8a557fa8152958c27485accdf409a23e
#owner: https://api.github.com/users/djun

# Other streamlit imports
-----------------------------------------
import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
application = get_wsgi_application()
------------------------------------------
# Rest of the Streamlit content