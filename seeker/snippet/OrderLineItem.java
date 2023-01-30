//date: 2023-01-30T16:48:20Z
//url: https://api.github.com/gists/0f34b168d02cee552e61f6839d52dcaf
//owner: https://api.github.com/users/ofarras

Product2 p = new Product2();
p.Name = 'Test Product';
p.Description='Test Product';
p.productCode = 'ABC';
p.isActive = true;
insert p;
		
PricebookEntry standardPrice = new PricebookEntry();
standardPrice.Pricebook2Id = Test.getStandardPricebookId();
standardPrice.Product2Id = p.Id;
standardPrice.UnitPrice = 100;
standardPrice.IsActive = true;
standardPrice.UseStandardPrice = false;
insert standardPrice ;

// insert account
Account acc = new Account(
	Name = 'SFDCPanther.com',
	Rating = 'Hot',
	Industry = 'Banking',
	Phone = '9087654321'
);
insert acc;

Order order = new Order(
	AccountId = acc.Id,
	EffectiveDate = System.today(),
	Status = 'Draft',
	PriceBook2Id = Test.getStandardPricebookId()
);
insert order;

OrderItem lineItem = new OrderItem();
lineItem.OrderId = order.id;
lineItem.Quantity = 24;
lineItem.UnitPrice = 240;
lineItem.Product2id = p.id;
lineItem.PricebookEntryId=standardPrice.id;
insert lineItem;

// Now update & Activate the Order
order.Status = 'Activated';
update order;