//date: 2022-01-25T17:05:30Z
//url: https://api.github.com/gists/fb1866c3e8a2fed5a2456ff4fca2a306
//owner: https://api.github.com/users/ragunathrajasekaran

@PutMapping(value = "/accounts/{accountId}")
private ResponseEntity<Account> updateAccount(@RequestBody Account account, @PathVariable Long accountId) {
    return this.accountRepository
            .findById(accountId)
            .map(accountFound -> {
                accountFound.mergeAccount(account);
                return ResponseEntity
                        .accepted()
                        .body(this.accountRepository.save(account));
            }).orElse(ResponseEntity.notFound().build());
}