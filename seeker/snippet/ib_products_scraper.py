#date: 2025-01-10T17:11:28Z
#url: https://api.github.com/gists/9cc0255a7428b59edc9cd43fd418a284
#owner: https://api.github.com/users/domkm

import polars as pl
import httpx

class IBProductsScraper:
    """
    https://www.interactivebrokers.com/en/trading/products-exchanges.php
    """
    
    def __init__(self):
        pass

    def _fetch(self, **kwargs) -> dict:
        """
        Takes in a dictionary of parameters and returns response.
        Params should include:
        pageNumber: int starting from 1
        productType: list of "FUT", "FOP", "IND", and/or "OPT"
        newProduct: "all", "T", or "F" (for whether to filter to products <= 30 days old)
        """
        url = "https://www.interactivebrokers.com/webrest/search/products-by-filters"
        data = {
            "pageNumber": 1,
            "pageSize": "500",
            "sortField": "symbol",
            "sortDirection": "asc",
            "productCountry": [],
            "productSymbol": "",
            "newProduct": "all",
            "productType": [],
            "domain": "com"
        }
        data.update(kwargs)
        print("Fetching", data)
        resp = httpx.post(url, headers={}, data=data)
        return resp.json()

    def _fetch_all(self, **kwargs) -> list:
        """
        Fetches all products using pagination.
        """
        params = {}
        params.update(kwargs)
        params["pageNumber"] = 1
        products = []
        while True:
            print("Fetching page", params["pageNumber"])
            resp = self._fetch(**params)
            products.extend(resp["products"])
            if len(products) >= resp["productCount"]:
                break
            params["pageNumber"] += 1
        return products
    
    def fetch_all(self, product: str, incremental: bool = False) -> pl.DataFrame:
        data = self._fetch_all(productType=[product], newProduct= "T" if incremental else "all")
        df = pl.DataFrame(data).unique()
        df.write_csv(f"product-{product}.csv")
        return df

df = IBProductsScraper().fetch_all("FUT")
