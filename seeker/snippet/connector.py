#date: 2026-01-23T16:58:59Z
#url: https://api.github.com/gists/ed9c28af03b46852c98f9a17065e773c
#owner: https://api.github.com/users/jonele

"""
Firebird database connector
"""
import logging
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import firebird-driver
try:
    from firebird.driver import connect, Connection, driver_config
    FIREBIRD_AVAILABLE = True

    # Configure Firebird client library location
    fb_paths = [
        os.environ.get('FIREBIRD'),
        r'C:\Program Files\Firebird\Firebird_3_0',
        r'C:\Program Files (x86)\Firebird\Firebird_3_0',
        r'C:\Program Files\Firebird\Firebird_4_0',
        r'C:\Program Files (x86)\Firebird\Firebird_4_0',
        r'C:\Firebird',
    ]

    for fb_path in fb_paths:
        if fb_path and os.path.exists(fb_path):
            lib_path = os.path.join(fb_path, 'fbclient.dll')
            if os.path.exists(lib_path):
                driver_config.fb_client_library.value = lib_path
                logger.info(f"Configured Firebird client: {lib_path}")
                break
    else:
        # Check if fbclient.dll is in current directory or agent folder
        local_paths = [
            'fbclient.dll',
            os.path.join(os.path.dirname(__file__), '..', '..', 'fbclient.dll'),
            r'C:\el-os\FBridge\fbclient.dll',
        ]
        for local_lib in local_paths:
            if os.path.exists(local_lib):
                driver_config.fb_client_library.value = os.path.abspath(local_lib)
                logger.info(f"Configured Firebird client (local): {local_lib}")
                break

except ImportError:
    FIREBIRD_AVAILABLE = False
    logger.warning("firebird-driver not installed")


class FirebirdConnector:
    """
    Firebird database connector with connection pooling
    """

    def __init__(self, database: "**********": str = "SYSDBA", password: str = "masterkey"):
        self.database = database
        self.user = user
        self.password = "**********"
        self.connection: Optional[Connection] = None

    async def connect(self) -> bool:
        """Connect to Firebird database"""
        if not FIREBIRD_AVAILABLE:
            logger.error("firebird-driver not available")
            return False

        try:
            self.connection = connect(
                database=self.database,
                user=self.user,
                password= "**********"
                charset="UTF8",
            )
            logger.info(f"Connected to Firebird: {self.database}")
            return True

        except Exception as e:
            logger.error(f"Firebird connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from Firebird")

    @contextmanager
    def cursor(self):
        """Get a cursor for executing queries"""
        if not self.connection:
            raise RuntimeError("Not connected to database")

        cur = self.connection.cursor()
        try:
            yield cur
        finally:
            cur.close()

    async def execute(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts"""
        with self.cursor() as cur:
            cur.execute(query, params or ())

            if cur.description:
                columns = [col[0] for col in cur.description]
                results = []
                for row in cur.fetchall():
                    # Decode CP737 bytes to strings
                    decoded_row = []
                    for val in row:
                        if isinstance(val, bytes):
                            try:
                                decoded_row.append(val.decode('cp737'))
                            except:
                                decoded_row.append(val.decode('utf-8', errors='replace'))
                        else:
                            decoded_row.append(val)
                    results.append(dict(zip(columns, decoded_row)))
                return results

            return []

    async def execute_write(self, query: str, params: tuple = None) -> int:
        """Execute a write query and return affected rows"""
        with self.cursor() as cur:
            cur.execute(query, params or ())
            self.connection.commit()
            return cur.rowcount

    def _decode(self, value) -> str:
        """Decode CP737 Greek text to UTF-8"""
        if value is None:
            return ''
        if isinstance(value, bytes):
            try:
                return value.decode('cp737')
            except:
                return value.decode('utf-8', errors='replace')
        return str(value)

    async def get_venues(self) -> List[Dict[str, Any]]:
        """Get all revenue centers (venues) from database"""
        try:
            # OREXSYS doesn't have ISACTIVE column - get all venues
            result = await self.execute(
                "SELECT RVCSID, NAME FROM RVCS"
            )
            return [{"rvcsid": r["RVCSID"], "name": self._decode(r["NAME"])} for r in result]
        except Exception as e:
            logger.error(f"Failed to get venues: {e}")
            return []
ecode(r["NAME"])} for r in result]
        except Exception as e:
            logger.error(f"Failed to get venues: {e}")
            return []
