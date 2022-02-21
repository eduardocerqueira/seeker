//date: 2022-02-21T16:53:44Z
//url: https://api.github.com/gists/9a12347b822e1f216a9e54beb920e1b3
//owner: https://api.github.com/users/donvadicastro

public class AuditTrailListener {
    @PrePersist
    public void beforeCreate(Object object) {
        log.info(createAudit(object, "CREATE"));
    }

    @PreUpdate
    private void beforeUpdate(Object object) {
        log.info(createAudit(object, "UPDATE"));
    }

    @PreRemove
    private void beforeRemove(Object object) {
        log.info(createAudit(object, "REMOVE"));
    }
}