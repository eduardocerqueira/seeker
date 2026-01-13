#date: 2026-01-13T17:16:49Z
#url: https://api.github.com/gists/f3c180957ab2a4cea91fd8895f5f6b73
#owner: https://api.github.com/users/vietthangif

"""
Superset API Client for creating datasets, charts, and dashboards.
"""
import os
from pathlib import Path
import requests
from typing import Optional, Dict, Any, Literal

class SupersetClient:
    """Client for interacting with Apache Superset REST API."""

    def __init__(self, verbose: bool = False):
        self.host = os.getenv("SUPERSET_HOST")
        self.username = os.getenv("SUPERSET_USERNAME")
        self.password = "**********"
        self.session = requests.Session()
        self.access_token: "**********"
        self.csrf_token: "**********"
        self.verbose = verbose
    
    def login(self) -> bool:
        """Authenticate with Superset and obtain access token."""
        login_url = f"{self.host}/api/v1/security/login"
        payload = {
            "username": self.username,
            "password": "**********"
            "provider": "db",
            "refresh": True
        }
        
        response = self.session.post(login_url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = "**********"
            self._get_csrf_token()
            return True
        else:
            if self.verbose:
                print(f"Login failed: {response.status_code} - {response.text}")
            else:
                print(f"Login failed: {response.status_code}")
            return False
    
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"g "**********"e "**********"t "**********"_ "**********"c "**********"s "**********"r "**********"f "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********"  "**********"- "**********"> "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
        """Get CSRF token for authenticated requests."""
        csrf_url = "**********"
        headers = self._get_headers()
        
        response = self.session.get(csrf_url, headers=headers)
        
        if response.status_code == 200:
            self.csrf_token = "**********"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            headers["Authorization"] = "**********"
        
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"s "**********"r "**********"f "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            headers["X-CSRFToken"] = "**********"
        
        return headers
    
    def create_sql_dataset(
        self,
        database_id: int,
        sql: str,
        table_name: str,
        schema: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Create a virtual dataset (SQL Lab query) in Superset.

        Args:
            database_id: ID of the database connection in Superset
            sql: SQL query for the virtual dataset
            table_name: Name for the virtual dataset
            schema: Database schema name

        Returns:
            Dataset response data or None if failed
        """
        url = f"{self.host}/api/v1/dataset/"

        payload = {
            "database": database_id,
            "table_name": table_name,
            "schema": schema,
            "sql": sql
        }

        response = self.session.post(url, json=payload, headers=self._get_headers())

        if response.status_code in [200, 201]:
            print(f"SQL Dataset '{table_name}' created successfully")
            return response.json()
        else:
            if self.verbose:
                print(f"Failed to create SQL dataset: {response.status_code} - {response.text}")
            else:
                print(f"Failed to create SQL dataset: {response.status_code}")
            return None
    
    def get_dataset_by_name(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get dataset by table name."""
        import json

        url = f"{self.host}/api/v1/dataset/"
        filters = {
            "filters": [
                {"col": "table_name", "opr": "eq", "value": table_name}
            ]
        }
        params = {"q": json.dumps(filters)}

        response = self.session.get(url, params=params, headers=self._get_headers())

        if response.status_code == 200:
            data = response.json()
            if data.get("count", 0) > 0:
                return data["result"][0]
        return None

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset by ID."""
        url = f"{self.host}/api/v1/dataset/{dataset_id}"

        response = self.session.delete(url, headers=self._get_headers())

        if response.status_code in [200, 204]:
            print(f"Dataset {dataset_id} deleted successfully")
            return True
        else:
            if self.verbose:
                print(f"Failed to delete dataset {dataset_id}: {response.status_code} - {response.text}")
            else:
                print(f"Failed to delete dataset {dataset_id}: {response.status_code}")
            return False

    def update_sql_dataset(
        self,
        dataset_id: int,
        sql: str,
        table_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing SQL dataset.

        Args:
            dataset_id: ID of the dataset to update
            sql: New SQL query for the dataset
            table_name: Optional new name for the dataset

        Returns:
            Updated dataset data or None if failed
        """
        url = f"{self.host}/api/v1/dataset/{dataset_id}"

        payload = {"sql": sql}
        if table_name:
            payload["table_name"] = table_name

        response = self.session.put(url, json=payload, headers=self._get_headers())

        if response.status_code == 200:
            print(f"Dataset {dataset_id} updated successfully")
            return response.json()
        else:
            if self.verbose:
                print(f"Failed to update dataset {dataset_id}: {response.status_code} - {response.text}")
            else:
                print(f"Failed to update dataset {dataset_id}: {response.status_code}")
            return None

    def create_chart(
        self,
        chart_name: str,
        chart_type: str,
        datasource_id: int,
        datasource_type: str = "table",
        params: Optional[Dict[str, Any]] = None,
        query_context: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new chart in Superset.
        
        Args:
            chart_name: Name of the chart
            chart_type: Type of chart (e.g., 'big_number_total', 'bar', 'pie')
            datasource_id: ID of the dataset to use
            datasource_type: Type of datasource (default: 'table')
            params: Chart parameters as a dict
            query_context: Query context for the chart
            description: Optional description
            
        Returns:
            Chart response data or None if failed
        """
        url = f"{self.host}/api/v1/chart/"
        
        import json
        
        payload = {
            "slice_name": chart_name,
            "viz_type": chart_type,
            "datasource_id": datasource_id,
            "datasource_type": datasource_type,
            "description": description,
            "params": json.dumps(params) if params else "{}",
        }
        
        if query_context:
            payload["query_context"] = json.dumps(query_context)
        
        response = self.session.post(url, json=payload, headers=self._get_headers())
        
        if response.status_code in [200, 201]:
            print(f"Chart '{chart_name}' created successfully")
            return response.json()
        else:
            if self.verbose:
                print(f"Failed to create chart: {response.status_code} - {response.text}")
            else:
                print(f"Failed to create chart: {response.status_code}")
            return None
    
    def create_dashboard(
        self,
        dashboard_title: str,
        slug: Optional[str] = None,
        published: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new dashboard in Superset.
        
        Args:
            dashboard_title: Title of the dashboard
            slug: URL slug for the dashboard (optional)
            published: Whether the dashboard is published
            
        Returns:
            Dashboard response data or None if failed
        """
        url = f"{self.host}/api/v1/dashboard/"
        
        payload = {
            "dashboard_title": dashboard_title,
            "published": published
        }
        
        if slug:
            payload["slug"] = slug
        
        response = self.session.post(url, json=payload, headers=self._get_headers())
        
        if response.status_code in [200, 201]:
            print(f"Dashboard '{dashboard_title}' created successfully")
            return response.json()
        else:
            if self.verbose:
                print(f"Failed to create dashboard: {response.status_code} - {response.text}")
            else:
                print(f"Failed to create dashboard: {response.status_code}")
            return None
    
    def get_dashboard_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get dashboard by slug."""
        url = f"{self.host}/api/v1/dashboard/"
        params = {
            "q": f'(filters:!((col:slug,opr:eq,value:{slug})))'
        }
        
        response = self.session.get(url, params=params, headers=self._get_headers())
        
        if response.status_code == 200:
            data = response.json()
            if data.get("count", 0) > 0:
                return data["result"][0]
        return None
    
    def add_chart_to_dashboard(
        self,
        dashboard_id: int,
        chart_ids: list,
        position_json: Optional[Dict[str, Any]] = None,
        json_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add charts to a dashboard.

        Args:
            dashboard_id: ID of the dashboard
            chart_ids: List of chart IDs to add
            position_json: Optional position configuration for chart layout
            json_metadata: Optional metadata including native filter configuration

        Returns:
            Updated dashboard data or None if failed
        """
        url = f"{self.host}/api/v1/dashboard/{dashboard_id}"

        import json

        # Build default position JSON if not provided
        if position_json is None:
            position_json = self._build_default_position(chart_ids)

        # Link each chart to the dashboard (required for association)
        for chart_id in chart_ids:
            self._link_chart_to_dashboard(chart_id, dashboard_id)

        payload = {
            "position_json": json.dumps(position_json)
        }

        # Add json_metadata for native filters if provided
        if json_metadata:
            payload["json_metadata"] = json.dumps(json_metadata)

        response = self.session.put(url, json=payload, headers=self._get_headers())
        
        if response.status_code == 200:
            print(f"Charts added to dashboard successfully")
            return response.json()
        else:
            if self.verbose:
                print(f"Failed to add charts to dashboard: {response.status_code} - {response.text}")
            else:
                print(f"Failed to add charts to dashboard: {response.status_code}")
            return None

    def _link_chart_to_dashboard(self, chart_id: int, dashboard_id: int) -> None:
        """Link a chart to a dashboard entity."""
        get_url = f"{self.host}/api/v1/chart/{chart_id}"
        put_url = f"{self.host}/api/v1/chart/{chart_id}"
        
        # Get current chart details
        response = self.session.get(get_url, headers=self._get_headers())
        if response.status_code != 200:
            print(f"Warning: Could not fetch chart {chart_id} to link to dashboard")
            return
            
        result = response.json().get("result", {})
        dashboards = result.get("dashboards", [])
        
        # Check if already linked using ID comparison
        # dashboards list usually contains objects with 'id' or just ids depending on endpoint
        # The GET /chart/{id} returns list of objects usually? 
        # Wait, debug output showed: Dashboards: []
        # If it returns objects, we need to extract IDs. 
        # If it returns IDs, we just append.
        # Let's handle both.
        
        current_ids = []
        for d in dashboards:
            if isinstance(d, dict):
                current_ids.append(d.get("id"))
            else:
                current_ids.append(d)
        
        if dashboard_id in current_ids:
            return
            
        current_ids.append(dashboard_id)
        
        payload = {
            "dashboards": current_ids
        }
        
        update_resp = self.session.put(put_url, json=payload, headers=self._get_headers())
        if update_resp.status_code == 200:
            print(f"Linked chart {chart_id} to dashboard {dashboard_id}")
        else:
            if self.verbose:
                print(f"Failed to link chart {chart_id} to dashboard: {update_resp.text}")
            else:
                print(f"Failed to link chart {chart_id} to dashboard")
    
    def _build_default_position(self, chart_ids: list) -> Dict[str, Any]:
        """Build default position JSON for dashboard layout."""
        position = {
            "DASHBOARD_VERSION_KEY": "v2",
            "ROOT_ID": {
                "type": "ROOT",
                "id": "ROOT_ID",
                "children": ["GRID_ID"]
            },
            "GRID_ID": {
                "type": "GRID",
                "id": "GRID_ID",
                "children": [],
                "parents": ["ROOT_ID"]
            },
            "HEADER_ID": {
                "id": "HEADER_ID",
                "type": "HEADER",
                "meta": {
                    "text": "Task Report Dashboard"
                }
            }
        }
        
        row_id = "ROW-1"
        position["GRID_ID"]["children"].append(row_id)
        position[row_id] = {
            "type": "ROW",
            "id": row_id,
            "children": [],
            "parents": ["ROOT_ID", "GRID_ID"],
            "meta": {
                "background": "BACKGROUND_TRANSPARENT"
            }
        }
        
        # Calculate width: 12 columns total, divide evenly among charts
        num_charts = len(chart_ids)
        chart_width = max(2, 12 // num_charts) if num_charts > 0 else 4

        for i, chart_id in enumerate(chart_ids):
            chart_key = f"CHART-{chart_id}"
            position[row_id]["children"].append(chart_key)
            position[chart_key] = {
                "type": "CHART",
                "id": chart_key,
                "children": [],
                "parents": ["ROOT_ID", "GRID_ID", row_id],
                "meta": {
                    "width": chart_width,
                    "height": 50,
                    "chartId": chart_id
                }
            }

        return position
    
    def get_databases(self) -> Optional[list]:
        """Get list of all database connections."""
        url = f"{self.host}/api/v1/database/"

        response = self.session.get(url, headers=self._get_headers())

        if response.status_code == 200:
            return response.json().get("result", [])
        else:
            if self.verbose:
                print(f"Failed to get databases: {response.status_code} - {response.text}")
            else:
                print(f"Failed to get databases: {response.status_code}")
            return None

    def get_chart_by_name(self, chart_name: str) -> Optional[Dict[str, Any]]:
        """Get chart by name."""
        import json

        url = f"{self.host}/api/v1/chart/"
        # Use JSON-encoded filter for proper handling of special characters
        filters = {
            "filters": [
                {"col": "slice_name", "opr": "eq", "value": chart_name}
            ]
        }
        params = {"q": json.dumps(filters)}

        response = self.session.get(url, params=params, headers=self._get_headers())

        if response.status_code == 200:
            data = response.json()
            if data.get("count", 0) > 0:
                return data["result"][0]
        return None

    def delete_chart(self, chart_id: int) -> bool:
        """Delete a chart by ID."""
        url = f"{self.host}/api/v1/chart/{chart_id}"

        response = self.session.delete(url, headers=self._get_headers())

        if response.status_code in [200, 204]:
            print(f"Chart {chart_id} deleted successfully")
            return True
        else:
            if self.verbose:
                print(f"Failed to delete chart {chart_id}: {response.status_code} - {response.text}")
            else:
                print(f"Failed to delete chart {chart_id}: {response.status_code}")
            return False

    def update_chart(
        self,
        chart_id: int,
        params: Dict[str, Any],
        datasource_id: int = None,
        chart_type: str = None,
        chart_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing chart.

        Args:
            chart_id: ID of the chart to update
            params: Chart parameters as a dict
            datasource_id: Optional new datasource ID
            chart_type: Optional new chart type (viz_type)
            chart_name: Optional new chart name

        Returns:
            Updated chart data or None if failed
        """
        import json

        url = f"{self.host}/api/v1/chart/{chart_id}"

        # Fetch existing chart so we can preserve required fields (viz_type, slice_name, datasource_type)
        existing = {}
        existing_resp = self.session.get(url, headers=self._get_headers())
        if existing_resp.status_code == 200:
            existing = existing_resp.json().get("result", {}) or {}
        else:
            print(f"Warning: Could not fetch chart {chart_id} before update ({existing_resp.status_code})")

        payload = {
            "params": json.dumps(params)
        }

        payload["datasource_id"] = datasource_id or existing.get("datasource_id")
        payload["viz_type"] = chart_type or existing.get("viz_type")
        payload["slice_name"] = chart_name or existing.get("slice_name")

        datasource_type = existing.get("datasource_type")
        if datasource_type:
            payload["datasource_type"] = datasource_type

        # Remove any None values to avoid API errors
        payload = {k: v for k, v in payload.items() if v is not None}

        if self.verbose:
            # Debug logging
            print(f"\n[DEBUG] === Update Chart {chart_id} ===")
            print(f"[DEBUG] URL: {url}")
            print(f"[DEBUG] Payload keys: {list(payload.keys())}")
            print(f"[DEBUG] datasource_id: {payload.get('datasource_id')}")
            params_str = json.dumps(params, indent=2)
            print(f"[DEBUG] Params (first 800 chars):\n{params_str[:800]}")

        response = self.session.put(url, json=payload, headers=self._get_headers())

        if self.verbose:
            # Debug response
            print(f"[DEBUG] Response status: {response.status_code}")
            print(f"[DEBUG] Response body: {response.text[:1500]}")

        if response.status_code == 200:
            print(f"Chart {chart_id} updated successfully")
            return response.json()
        else:
            if self.verbose:
                print(f"Failed to update chart {chart_id}: {response.status_code} - {response.text}")
            else:
                print(f"Failed to update chart {chart_id}: {response.status_code}")
            return None

    def delete_dashboard(self, dashboard_id: int) -> bool:
        """Delete a dashboard by ID."""
        url = f"{self.host}/api/v1/dashboard/{dashboard_id}"

        response = self.session.delete(url, headers=self._get_headers())

        if response.status_code in [200, 204]:
            print(f"Dashboard {dashboard_id} deleted successfully")
            return True
        else:
            if self.verbose:
                print(f"Failed to delete dashboard {dashboard_id}: {response.status_code} - {response.text}")
            else:
                print(f"Failed to delete dashboard {dashboard_id}: {response.status_code}")
            return False

    def update_dashboard_filters(
        self,
        dashboard_id: int,
        filters: list,
        deleted: list = None,
        reordered: list = None,
        orientation: Literal["HORIZONTAL", "VERTICAL"] = "HORIZONTAL"
    ) -> Optional[Dict[str, Any]]:
        """
        Update native filters on a dashboard using the dedicated filters endpoint.

        Args:
            dashboard_id: ID of the dashboard
            filters: List of filter configurations to add/update
            deleted: List of filter IDs to delete
            reordered: List of filter IDs in desired order

        Returns:
            Updated dashboard data or None if failed
        """
        url = f"{self.host}/api/v1/dashboard/{dashboard_id}/filters"

        payload = {
            "deleted": deleted or [],
            "modified": filters,
            "reordered": reordered or [f["id"] for f in filters]
        }

        response = self.session.put(url, json=payload, headers=self._get_headers())

        if response.status_code == 200:
            print(f"Dashboard {dashboard_id} filters updated successfully")
            self._update_dashboard_metadata_for_filters(
                dashboard_id=dashboard_id,
                filters=filters,
                orientation=orientation
            )
            return response.json()
        else:
            if self.verbose:
                print(f"Failed to update dashboard filters: {response.status_code} - {response.text}")
            else:
                print(f"Failed to update dashboard filters: {response.status_code}")
            return None

    def _update_dashboard_metadata_for_filters(
        self,
        dashboard_id: int,
        filters: list,
        orientation: str
    ) -> None:
        """
        Sync filter configuration into dashboard json_metadata for embedding.
        """
        import json

        get_url = f"{self.host}/api/v1/dashboard/{dashboard_id}"
        resp = self.session.get(get_url, headers=self._get_headers())

        if resp.status_code != 200:
            print(f"Warning: Could not fetch dashboard {dashboard_id} metadata to sync filters")
            return

        result = resp.json().get("result", {})
        metadata_raw = result.get("json_metadata") or "{}"

        try:
            metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        except json.JSONDecodeError:
            metadata = {}

        if metadata is None:
            metadata = {}

        # Ensure native filter configuration and scopes are present for embed clients
        metadata["native_filter_configuration"] = filters
        metadata["filter_scopes"] = {
            f["id"]: {
                "scope": f.get("scope", {}).get("rootPath", ["ROOT_ID"]),
                "immune": f.get("scope", {}).get("excluded", [])
            } for f in filters
        }
        metadata["filter_bar_orientation"] = orientation

        payload = {"json_metadata": json.dumps(metadata)}
        update_resp = self.session.put(get_url, json=payload, headers=self._get_headers())

        if update_resp.status_code == 200:
            print(f"Dashboard {dashboard_id} json_metadata updated with filters/orientation")
        else:
            if self.verbose:
                print(f"Warning: Failed to update dashboard metadata: {update_resp.status_code} - {update_resp.text}")
            else:
                print(f"Warning: Failed to update dashboard metadata: {update_resp.status_code}")
