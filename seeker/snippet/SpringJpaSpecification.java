//date: 2022-03-31T17:08:17Z
//url: https://api.github.com/gists/0039391600b3a1fbfbfd03ce48d2776f
//owner: https://api.github.com/users/pkkwilliam

// Item contain List<Category> find all items by Categories id
protected Specification<Item> equalCategoryId(ItemQueryRequest itemQueryRequest) {
  return itemQueryRequest.getCategoryId() == null ? null : (root, criteriaQuery, criteriaBuilder) -> criteriaBuilder.equal(root.join("categories").get("id"), itemQueryRequest.getCategoryId());
}