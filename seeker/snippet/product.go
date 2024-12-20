//date: 2024-12-20T16:55:54Z
//url: https://api.github.com/gists/ca4bbc67742e9adca587643ac1ca2b5f
//owner: https://api.github.com/users/Muhafiz22

func (p *Product) Get(Products []Product) { //to Get Product details
	//It Accepts Products slice to check if the product to be added is already in the inventory's product list

	fmt.Print("Enter Product ID:")
	if _, err := fmt.Scan(&p.ID); err != nil {
		fmt.Print(p.ID)
		fmt.Println(" Invalid input, Please Enter valid ID")
	}

	existingProduct := FindProductById(Products, p.ID) // to find if same Product is already exixting in inventory or not

	if existingProduct != nil { // if same Product is already existing in inventory then add in existing Product details

		fmt.Print("Enter additional Quantity:")
		var QdditionalQuantity int
		fmt.Scan(&QdditionalQuantity)

		existingProduct.Quantity += QdditionalQuantity
		p.ID = existingProduct.ID
		p.Name = existingProduct.Name
		p.Price = existingProduct.Price
		p.Quantity = existingProduct.Quantity
	} else { // adding new Product to inventory

		fmt.Println("Enter Product Name:")
		fmt.Scan(&p.Name)
		fmt.Println("Enter Product Price:")
		fmt.Scan(&p.Price)
		fmt.Println("Enter Product Quantity:")
		fmt.Scan(&p.Quantity)
	}
}