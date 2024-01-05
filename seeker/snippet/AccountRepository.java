//date: 2024-01-05T17:06:11Z
//url: https://api.github.com/gists/1ebfddc16f8f4ba29f32339660264f7f
//owner: https://api.github.com/users/Ribeiro

package br.com.stackspot.nullbank.withdrawal;

import org.hibernate.LockOptions;
import org.springframework.data.jpa.repository.*;
import org.springframework.stereotype.Repository;

import javax.persistence.LockModeType;
import javax.persistence.QueryHint;
import javax.transaction.Transactional;
import java.util.Optional;

@Repository
public interface AccountRepository extends JpaRepository<Account, Long> {

    /**
     * Loads the entity even when a lock is not acquired
     */ 
    @Transactional
    @Query(value = """
                   select new br.com.stackspot.nullbank.withdrawal.LockableAccount(
                          c
                         ,pg_try_advisory_xact_lock(c.id)
                         )
                     from Account c
                    where c.id = :accountId
                      and pg_try_advisory_xact_lock(c.id) is not null
                   """
    )
    public Optional<LockableAccount> findByIdWithPessimisticAdvisoryLocking(Long accountId);

}
