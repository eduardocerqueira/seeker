#date: 2024-05-15T17:00:10Z
#url: https://api.github.com/gists/cfb11e467ff671f4ccfb2cd233909034
#owner: https://api.github.com/users/charotAmine

def createTable_productSupplier(environment):
    print(f'Creating product_supplier table in {environment}_catalog')
    spark.sql(f"""CREATE TABLE IF NOT EXISTS `{environment}_catalog`.`bronze`.`product_supplier`
                        (
                            ProductID BIGINT,
                            ProductLine VARCHAR(50),
                            ProductCategory VARCHAR(50),
                            ProductGroup VARCHAR(50),
                            ProductName VARCHAR(100),
                            SupplierCountry VARCHAR(2),
                            SupplierName VARCHAR(100),
                            SupplierID INT
                    );""")
    
    print("************************************")

def createTable_customerOrders(environment):
    print(f'Creating customer_orders table in {environment}_catalog')
    spark.sql(f"""CREATE TABLE IF NOT EXISTS `{environment}_catalog`.`bronze`.`customer_orders`
                        (
                            CustomerID INT,
                            CustomerStatus VARCHAR(20),
                            DateOrderPlaced DATE,
                            DeliveryDate DATE,
                            OrderID INT,
                            ProductID BIGINT,
                            QuantityOrdered INT,
                            TotalRetailPrice DOUBLE,
                            CostPricePerUnit DOUBLE
                    );""")
    
    print("************************************")

createTable_productSupplier(env)
createTable_customerOrders(env)