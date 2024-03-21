#date: 2024-03-21T16:59:58Z
#url: https://api.github.com/gists/1280a47d1a88354f71dc85643787a78b
#owner: https://api.github.com/users/Djama1GIT

from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, lit


# Класс для работы с продуктами и категориями
class ProductCategory:
    def __init__(self, products_df, categories_df, connections_df):
        # Инициализация DataFrame для продуктов, категорий и связей между ними
        self.products_df = products_df
        self.categories_df = categories_df
        self.connections_df = connections_df

    # Метод для вывода пар продукт-категория с учетом связей
    def print_product_category_pairs(self):
        # Создание пар продукт-категория через перекрестное соединение
        product_category_pairs = self.products_df.select("product_id", "product_name").crossJoin(
            self.categories_df.select("category_id", "category_name"))

        # Присоединение данных о связях
        result_with_connections = self.join_with_connections(product_category_pairs)

        # Вывод результата
        self.display_result("Product-Category Pairs with Connections:", result_with_connections)

        # Поиск продуктов без категорий
        products_without_categories = self.find_products_without_categories()

        # Вывод продуктов без категорий
        self.display_result("Products without Categories:", products_without_categories.select("product_name"))

    # Метод для присоединения пар продукт-категория с данными о связях
    def join_with_connections(self, product_category_pairs):
        return product_category_pairs.join(
            self.connections_df,
            on=["product_id", "category_id"],
            how="inner"
        ).select(
            concat(col("product_name"), lit(" – "), col("category_name")).alias("Product-Category")
        )

    # Метод для поиска продуктов без категорий
    def find_products_without_categories(self):
        return self.products_df.join(
            self.connections_df,
            on="product_id",
            how="left_outer"
        ).filter(self.connections_df["category_id"].isNull())

    # Метод для вывода результата
    @staticmethod
    def display_result(title, df):
        print(title)
        df.show()


# Создание сессии Spark
spark = SparkSession.builder.appName("ProductCategoryPairs").getOrCreate()

# Данные о продуктах
products_data = [("1", "Product1"), ("2", "Product2"), ("3", "Product3")]
# Данные о категориях
categories_data = [("1", "Category1"), ("2", "Category2"), ("3", "Category3")]
# Данные о связях между продуктами и категориями
connections_data = [("1", "1"), ("2", "2"), ("3", None)]

# Создание DataFrame для продуктов
products_df = spark.createDataFrame(products_data, ["product_id", "product_name"])
# Создание DataFrame для категорий
categories_df = spark.createDataFrame(categories_data, ["category_id", "category_name"])
# Создание DataFrame для связей
connections_df = spark.createDataFrame(connections_data, ["product_id", "category_id"])

# Создание экземпляра класса ProductCategory
pc = ProductCategory(products_df, categories_df, connections_df)
# Вывод пар продукт-категория с учетом связей
pc.print_product_category_pairs()
