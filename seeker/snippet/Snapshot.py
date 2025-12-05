#date: 2025-12-05T17:05:37Z
#url: https://api.github.com/gists/3a6b3eb0f70c19b3dd8d89cf3006055a
#owner: https://api.github.com/users/manuelep

import os
import shutil
from datetime import datetime
from qgis.core import QgsProject

project_path = QgsProject.instance().fileName()

if not project_path:
    print("⚠ Nessun progetto aperto.")
else:
    gpkg_path = None

    # verifica se progetto è contenuto in un geopackage
    print(f'Path del progetto: {project_path}')
    if project_path.startswith("geopackage:"):
        # rimuove "geopackage:" e tutto dopo ?
        stripped = project_path.replace("geopackage:", "")
        gpkg_path = stripped.split("?")[0]
    else:
        print("❗ Il progetto non sembra essere salvato in un GeoPackage.")
    
    if gpkg_path:
        print(f'Path del geopackage: {gpkg_path}')
        base_dir = os.path.dirname(gpkg_path)
        print(base_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(base_dir, f"backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)

        backup_path = os.path.join(backup_dir, os.path.basename(gpkg_path))
        shutil.copy2(gpkg_path, backup_path)

        print(f"✔ Backup creato:\n{backup_path}")
