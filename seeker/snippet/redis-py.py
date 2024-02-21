#date: 2024-02-21T17:02:10Z
#url: https://api.github.com/gists/aa5a7653404d63771abda171b78a3e91
#owner: https://api.github.com/users/mvandermeulen

import json
from typing import List, Dict

from redis.commands.search import Search
from redis.commands.search.field import TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class RediJsonIo:
    def __init__(
        self,
        basic_schema_for_indexing,
        search_prefix_name,
        host="127.0.0.1",
        port=6379,
    ):
        import redis

        self.redis_connection = redis.Redis(host=host, port=port, decode_responses=True)
        self.search_prefix_name = search_prefix_name
        self.ft_name = f"idx:{search_prefix_name}"
        self.basic_schema = basic_schema_for_indexing

        self.__create_index()

    def __create_index(self) -> Search:
        try:
            self.redis_connection.ft(self.ft_name).info()
        except:
            index = self.redis_connection.ft(self.ft_name)
            return index.create_index(
                self.basic_schema,
                skip_initial_scan=True,
                definition=IndexDefinition(
                    prefix=[f"{self.search_prefix_name}:"],
                    index_type=IndexType.JSON,
                ),
            )

    def create(self, identifier, serialized_data):
        key = f"{self.search_prefix_name}:{identifier}"
        self.redis_connection.json().set(key, ".", serialized_data)

    def removed(self, identifier):
        key = f"{self.search_prefix_name}:{identifier}"
        self.redis_connection.delete(key)

    def search(
        self,
        search_key_value: List[Dict[str, str]],
        exclude_field=None,
        sort_by=None,
    ):
        """
        Perform a dynamic search based on various criteria.
        @search_key_value example:
        [
            {'name': '%napa extra 50%'}, # wildcard search
            {'brand': 'napa*'}, # ^text means start with and text^ means ends with
            {'price': '<100'}, # less than 100
            {'price': '>50'}, # greater than 50
        ]
        @exclude_field = ['uid', 'id']
        @sort_by {"field_name": ascending_true_or_false}
        """
        if sort_by is None:
            sort_by = {}
        if exclude_field is None:
            exclude_field = []

        index = self.redis_connection.ft(self.ft_name)
        query_str = ""

        # Building the query string
        for search_key in search_key_value:
            for key, value in search_key.items():
                # Construct query based on the type of search
                if value.startswith("%%%") and value.endswith("%%%"):
                    import re

                    cleaned_message = re.sub(r"[^a-zA-Z0-9\s]", "", value)

                    # Split the product name into separate terms
                    terms = cleaned_message.split()

                    # Construct a query string with wildcards for each term
                    query_str = " ".join([f"@{key}:%%%{term}%%%" for term in terms])

                elif value.startswith("%%") and value.endswith("%%"):
                    import re

                    cleaned_message = re.sub(r"[^a-zA-Z0-9\s]", "", value)

                    # Split the product name into separate terms
                    terms = cleaned_message.split()

                    # Construct a query string with wildcards for each term
                    query_str = " ".join([f"@{key}:%%{term}%%" for term in terms])

                elif value.startswith("%") and value.endswith("%"):
                    import re

                    cleaned_message = re.sub(r"[^a-zA-Z0-9\s]", "", value)
                    query_str += f" @{key}:{cleaned_message}"

                elif value.endswith("*"):
                    import re

                    cleaned_message = re.sub(r"[^a-zA-Z0-9\s]", "", value)
                    query_str += f" @{key}:{cleaned_message}*"

                elif value.startswith("*"):
                    import re

                    cleaned_message = re.sub(r"[^a-zA-Z0-9\s]", "", value)
                    query_str += f" @{key}:*{cleaned_message}"

                elif value.startswith("<"):
                    import re

                    cleaned_message = re.sub(r"[^a-zA-Z0-9\s]", "", value)
                    query_str += f" @{key}:[-inf {cleaned_message}]"

                elif value.startswith(">"):
                    import re

                    cleaned_message = re.sub(r"[^a-zA-Z0-9\s]", "", value)
                    query_str += f" @{key}:[({cleaned_message} +inf]"

                else:
                    query_str += f" @{key}:{value}"

        query = Query(query_str.strip())

        # Creating the query object
        if len(exclude_field) > 0:
            query = query.return_field(*exclude_field)

        if bool(sort_by):
            value_sort = sort_by.popitem()
            query = query.sort_by(value_sort[0], value_sort[1])

        # Performing the search
        results = index.search(query)
        final_list = [json.loads(result.__dict__["json"]) for result in results.docs]
        final_list.reverse()
        return final_list[:40]

    def search_wildcard(self, search_name, sort_by: Dict[str, bool] = None):
        # Split the product name into separate terms
        terms = search_name.split()

        # Construct a query string with wildcards for each term
        query_str = " ".join([f"@name:%%{term}%%" for term in terms])

        query = Query(query_str)
        if sort_by:
            """
            @sort_by {"name": ascending_true_or_false}
            """
            query.sort_by(sort_by[0], sort_by[1])

        # Execute the query
        return self.redis_connection.ft(self.ft_name).search(query)

    def update(self, identifier, serialized_data):
        self.removed(identifier=identifier)

        self.create(identifier=identifier, serialized_data=serialized_data)
