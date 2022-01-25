//date: 2022-01-25T16:52:40Z
//url: https://api.github.com/gists/1a16fdc1b4be44b8c3830e7d5586902d
//owner: https://api.github.com/users/ragunathrajasekaran

@GetMapping(value = "/accounts")
private ResponseEntity<Page<Account>> accounts(Pageable pageable) {
    return ResponseEntity
            .ok()
            .body(this.accountRepository.findAll(pageable));
}
@GetMapping(value = "/accounts/{accountId}")
private ResponseEntity<Account> accountById(@PathVariable Long accountId) {
    return this.accountRepository
            .findById(accountId)
            .map(ResponseEntity.accepted()::body)
            .orElse(ResponseEntity.notFound().build());
}