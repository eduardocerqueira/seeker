//date: 2022-01-25T17:12:04Z
//url: https://api.github.com/gists/a202befdc9caed18bf28a6691512f88e
//owner: https://api.github.com/users/ragunathrajasekaran

@DeleteMapping(value = "/accounts/{accountId}")
private ResponseEntity deleteAccount(@PathVariable Long accountId) {
    return this.accountRepository
            .findById(accountId)
            .map(accountFound -> {
                this.accountRepository.delete(accountFound);
                return ResponseEntity
                        .accepted()
                        .build();
            }).orElse(ResponseEntity.notFound().build());
}