#date: 2023-07-27T16:58:27Z
#url: https://api.github.com/gists/af47e60276c794ea45888cdb8e77f89b
#owner: https://api.github.com/users/Karatelee

from fastapi import FastAPI, HTTPException
from models.product import Product
import sqlite3

from models.productInput import ProductInput
app = FastAPI()

def mappingProduct(products: list):
    result = []
    for product in products:
        result.append(Product(product[0], product[1], product[2]))
    return result

@app.get("/products")
async def get_products(min_price: int = -1000, max_price: int = 1000):
    products = []
    with sqlite3.connect("eshop.db") as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id, title, price FROM products WHERE price BETWEEN ? AND ?", (min_price, max_price))
        products = cursor.fetchall()
    return mappingProduct(products)

@app.get("/products/{product_id}")
async def get_product(product_id: int):
    product = None
    with sqlite3.connect("eshop.db") as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id, title, price FROM products WHERE id = ?", (product_id,))
        product = cursor.fetchone()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return Product(product[0], product[1], product[2])

@app.delete("/products/{product_id}", status_code=204)
async def delete_product(product_id: int):
    with sqlite3.connect("eshop.db") as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM products WHERE id = ?", (product_id,))
    return {"message": "Product deleted"}

@app.post("/products", status_code=201)
async def create_product(product: ProductInput):
    with sqlite3.connect("eshop.db") as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO products (title, price) VALUES (?, ?)", (product.title, product.price))
    return {"message": "Product created"}

@app.put("/products/{product_id}", status_code=200)
async def update_product(product_id: int, product: ProductInput):
    with sqlite3.connect("eshop.db") as connection:
        cursor = connection.cursor()
        cursor.execute("UPDATE products SET title = ?, price = ? WHERE id = ?", (product.title, product.price, product_id))
    return {"message": "Product updated"}