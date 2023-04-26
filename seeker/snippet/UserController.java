//date: 2023-04-26T16:48:15Z
//url: https://api.github.com/gists/815391c5102da89876af5130df69eeb5
//owner: https://api.github.com/users/CodeVaDOs

package com.datingon.controller;

import com.datingon.dto.rq.MessageRequest;
import com.datingon.dto.rq.UpdateAvatarRequest;
import com.datingon.dto.rq.UpdateProfileRequest;
import com.datingon.dto.rs.UserResponse;
import com.datingon.facade.UserFacade;
import com.datingon.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.security.Principal;
import java.util.List;

@RequestMapping("api/v1/user")
@Validated
@RestController
public class UserController {
    private final UserFacade userFacade;
    private final UserService userService;

    public UserController(UserFacade userFacade, UserService userService) {
        this.userFacade = userFacade;
        this.userService = userService;
    }

    @GetMapping("profile")
    @PreAuthorize("hasAuthority('read')")
    public ResponseEntity<UserResponse> getProfile(Principal principal) {
        return ResponseEntity.ok(userFacade.getProfile(principal.getName()));
    }

    @GetMapping("suggestions")
    @PreAuthorize("hasAuthority('read')")
    public ResponseEntity<List<UserResponse>> getSuggestions(Principal principal) {
        return ResponseEntity.ok(userFacade.findAllSuggestionsByEmail(principal.getName()));
    }



    @GetMapping("matches")
    @PreAuthorize("hasAuthority('read')")
    public ResponseEntity<List<UserResponse>> getMatches(Principal principal) {
        return ResponseEntity.ok(userFacade.findAllMatchesByEmail(principal.getName()));
    }

    @PutMapping
    @PreAuthorize("hasAuthority('read')")
    public ResponseEntity<UserResponse> updateProfile(@RequestBody @Valid UpdateProfileRequest request) {
        return ResponseEntity.ok(userFacade.updateProfile(request));
    }

    @PutMapping("avatar")
    @PreAuthorize("hasAuthority('read')")
    public ResponseEntity<UserResponse> updateAvatar(@RequestBody @Valid UpdateAvatarRequest request) {
        return ResponseEntity.ok(userFacade.updateAvatar(request));
    }
}
