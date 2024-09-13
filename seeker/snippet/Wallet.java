//date: 2024-09-13T16:50:04Z
//url: https://api.github.com/gists/945ee148356f3d8d5c7529670c678342
//owner: https://api.github.com/users/dmzDAWG

class LedgerItem {
  InternalAccount account;

  static LedgerItem repayment(Repayment repayment) {
    return // LedgerItem().builder()...
  }
}

class InternalAccount {
  List<LedgerItem> ledgerItems;
  // account doesn't directly reference repayments
  // it just manages ledger items and represents a repayment as a ledger item

  void record(Repayment repayment) {
    ledgerItems.add(LedgerItem.repayment(repayment));
  }
}

class Repayment {
  InternalAccount internalAccount;

  public Repayment(InternalAccount internalAccount, Money amount) {
    this.internalAccount = internalAccount;
    internalAccount.record(this);
  }
}