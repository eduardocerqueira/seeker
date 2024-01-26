//date: 2024-01-26T16:42:44Z
//url: https://api.github.com/gists/522a0a2295abdbffe524cab337f19e3c
//owner: https://api.github.com/users/chroakPRO

package com.example.rbac;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.*;

import java.util.*;

// === Domain Classes ===
class User {
    private Long id;
    private String name;
    private String username;
    private String password;
    private Set<String> roles;

    // Constructors
    public User(Long id, String name, String username, String password, Set<String> roles) {
        this.id = id;
        this.name = name;
        this.username = username;
        this.password = "**********"
        this.roles = roles;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = "**********"
    public Set<String> getRoles() { return roles; }
    public void setRoles(Set<String> roles) { this.roles = roles; }
}

class RBACPermissionContainer {
    private RBACPermissionSet permissionSet;
    private boolean granted;
    private int permissionId;
    private User user;

    // Constructors
    public RBACPermissionContainer(RBACPermissionSet permissionSet, boolean granted, int permissionId, User user) {
        this.permissionSet = permissionSet;
        this.granted = granted;
        this.permissionId = permissionId;
        this.user = user;
    }

    // Getters and Setters
    public RBACPermissionSet getPermissionSet() { return permissionSet; }
    public void setPermissionSet(RBACPermissionSet permissionSet) { this.permissionSet = permissionSet; }
    public boolean isGranted() { return granted; }
    public void setGranted(boolean granted) { this.granted = granted; }
    public int getPermissionId() { return permissionId; }
    public void setPermissionId(int permissionId) { this.permissionId = permissionId; }
    public User getUser() { return user; }
    public void setUser(User user) { this.user = user; }
}

class RBAC {
    private User user;
    private Set<RBACPermissionContainer> permissionContainer;

    // Constructors
    public RBAC(User user, Set<RBACPermissionContainer> permissionContainer) {
        this.user = user;
        this.permissionContainer = permissionContainer;
    }

    // Getters and Setters
    public User getUser() { return user; }
    public void setUser(User user) { this.user = user; }
    public Set<RBACPermissionContainer> getPermissionContainer() { return permissionContainer; }
    public void setPermissionContainer(Set<RBACPermissionContainer> permissionContainer) { this.permissionContainer = permissionContainer; }
}

// === Enums ===
enum RBACCommandResults {
    RBAC_OK, RBAC_ERROR, RBAC_ERROR_INVALID_COMMAND, RBAC_ERROR_INVALID_ARGUMENT,
}

enum RBACCommands {
    RBAC_COMMAND_HAS_PERMISSION, RBAC_COMMAND_GRANT_PERMISSION, RBAC_COMMAND_REVOKE_PERMISSION,
}

enum RBACPermissionSet {
    RBAC_PERM_READ, RBAC_PERM_WRITE, RBAC_PERM_DELETE, RBAC_PERM_CREATE, RBAC_PERM_UPDATE, RBAC_PERM_EXECUTE,
}

// === Repository ===
@Repository
class RBACRepo {
    private final Map<String, RBAC> rbacMap = new HashMap<>();

    // Initializes with two example users
    public RBACRepo() {
        User user1 = "**********"
        User user2 = "**********"

        Set<RBACPermissionContainer> permissions1 = new HashSet<>();
        permissions1.add(new RBACPermissionContainer(RBACPermissionSet.RBAC_PERM_READ, true, 1, user1));
        rbacMap.put(user1.getUsername(), new RBAC(user1, permissions1));

        Set<RBACPermissionContainer> permissions2 = new HashSet<>();
        permissions2.add(new RBACPermissionContainer(RBACPermissionSet.RBAC_PERM_WRITE, true, 2, user2));
        rbacMap.put(user2.getUsername(), new RBAC(user2, permissions2));
    }

    public Optional<RBAC> findByUsername(String username) {
        return Optional.ofNullable(rbacMap.get(username));
    }
}

// === Service ===
@Service
class RBACService {
    @Autowired
    private RBACRepo repo;

    public boolean hasPermission(String username, RBACPermissionSet permission) {
        return repo.findByUsername(username)
                .map(rbac -> rbac.getPermissionContainer().stream()
                        .anyMatch(container -> container.getPermissionSet() == permission && container.isGranted()))
                .orElse(false);
    }
}

// === Controller ===
@RestController
@RequestMapping("/rbac")
class RBACResource {
    @Autowired
    private RBACService rbacService;

    @GetMapping("/test")
    public String testPermission(@RequestParam String username, @RequestParam RBACPermissionSet permission) {
        if (rbacService.hasPermission(username, permission)) {
            return "User has permission";
        } else {
            return "User does not have permission";
        }
    }
}
(username, permission)) {
            return "User has permission";
        } else {
            return "User does not have permission";
        }
    }
}
