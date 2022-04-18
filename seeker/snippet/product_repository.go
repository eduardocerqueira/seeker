//date: 2022-04-18T17:09:09Z
//url: https://api.github.com/gists/ca4de6d2c9c3b29dfb89fe00e4a287b8
//owner: https://api.github.com/users/rzldimam28

type ProductRepository struct {
	DB *sql.DB
}

// list of method
func (repository *ProductRepository) Save(product domain.Product) domain.Product {
	SQL := "INSERT INTO products (name) VALUES (?)"
	result, err := repository.DB.Exec(SQL, product.Name)
	helper.PanicIfError(err)

	id, err := result.LastInsertId()
	helper.PanicIfError(err)

	product.Id = int(id)
	return product
}

func (repository *ProductRepository) List() []domain.Product {
	SQL := "SELECT id, name FROM products"
	rows, err := repository.DB.Query(SQL)
	helper.PanicIfError(err)
	defer rows.Close()

	var products []domain.Product
	for rows.Next() {
		var product domain.Product
		err := rows.Scan(&product.Id, &product.Name)
		helper.PanicIfError(err)
		products = append(products, product)
	}
	return products
}