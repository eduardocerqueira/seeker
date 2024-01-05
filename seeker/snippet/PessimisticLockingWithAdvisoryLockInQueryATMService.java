//date: 2024-01-05T17:06:11Z
//url: https://api.github.com/gists/1ebfddc16f8f4ba29f32339660264f7f
//owner: https://api.github.com/users/Ribeiro

package br.com.stackspot.nullbank.withdrawal;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class PessimisticLockingWithAdvisoryLockInQueryATMService {

    @Autowired
    private AccountRepository repository;
    @Autowired
    private TransactionRepository transactionRepository;

    @Retryable(
            value = FailedToAcquireLockForAccountException.class,
            maxAttempts = 3,
            backoff = @Backoff(delay = 100, random = true, multiplier = 2.0)
    )
    @Transactional
    public void withdraw(Long accountId, double amount) {

        // We load the entity even if a lock is not acquired
        LockableAccount lockedAccount = repository.findByIdWithPessimisticAdvisoryLocking(accountId).orElseThrow(() -> {
            throw new IllegalStateException("account does not exist: " + accountId);
        });

        // But the business logic is executed only if the lock was acquired for the account
        Account account = lockedAccount
                .getAccountIfLockedOrElseThrow();

        double newBalance = (account.getBalance() - amount);
        if (newBalance < 0) {
            throw new IllegalStateException("there's not enough balance");
        }

        account.setBalance(newBalance);
        repository.save(account);

        transactionRepository
                .save(new Transaction(account, amount, "withdraw"));
    }

}

/**
 * Represents an account that may be locked or not
 */
class LockableAccount {
    private Account account;
    private boolean locked;

    public LockableAccount(Account account, boolean locked) {
        this.account = account;
        this.locked = locked;
    }

    /**
     * Returns the actual account if it was locked or else throws an {@code AccountNotFoundOrLockNotAcquiredException}
     */
    public Account getAccountIfLockedOrElseThrow() {
        if (!locked) {
            throw new FailedToAcquireLockForAccountException("Account already locked by another user");
        }
        return account;
    }
    
    public boolean isLocked() {
        return locked;
    }
}

class FailedToAcquireLockForAccountException extends RuntimeException {

    public FailedToAcquireLockForAccountException(String message) {
        super(message);
    }
}


