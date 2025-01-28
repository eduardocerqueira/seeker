//date: 2025-01-28T16:59:27Z
//url: https://api.github.com/gists/91d60a32522c858294fe76b224981f26
//owner: https://api.github.com/users/kcdevdes

package com.kcdevdes.poppick.controller;

import com.kcdevdes.poppick.dto.request.OauthSignupRequestDto;
import com.kcdevdes.poppick.dto.request.RefreshTokenRequestDto;
import com.kcdevdes.poppick.dto.response.UserResponseDto;
import com.kcdevdes.poppick.dto.response.JwtResponseDto;
import com.kcdevdes.poppick.dto.request.LoginRequestDto;
import com.kcdevdes.poppick.dto.request.SignupRequestDto;
import com.kcdevdes.poppick.service.AuthService;
import com.kcdevdes.poppick.service.UserService;
import com.kcdevdes.poppick.util.UserMapper;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;

import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping("/v1/auth")
public class AuthController {

    private final UserService userService;
    private final UserMapper userMapper = new UserMapper();
    private final AuthService authService;

    public AuthController(UserService userService, AuthService authService) {
        this.userService = userService;
        this.authService = authService;
    }

    /////////////////////////////////////////////////////////////
    //////////////// Standard Login /////////////////////////////
    /////////////////////////////////////////////////////////////

    @PostMapping("/signup")
    public ResponseEntity<UserResponseDto> signup(@Valid @RequestBody SignupRequestDto requestDto) {
        return ResponseEntity.status(HttpStatus.CREATED).body(userMapper.toDto(authService.registerUser(requestDto)));
    }

    @PostMapping("/login")
    public ResponseEntity<JwtResponseDto> login(@Valid @RequestBody LoginRequestDto requestDto) {
        return ResponseEntity.ok(authService.issueJWT(requestDto));
    }

    @PostMapping("/refresh")
    public ResponseEntity<JwtResponseDto> refreshAccessToken(@RequestHeader("Authorization") String token) {
        return ResponseEntity.ok(authService.refreshJWT(new RefreshTokenRequestDto(token)));
    }

    //////////////////////////////////////////////////////////////
    /////////////////   OAuth Login  /////////////////////////////
    //////////////////////////////////////////////////////////////

    @GetMapping("/google/login")
    public void loginWithGoogle(HttpServletResponse response) throws IOException {
        response.sendRedirect("/oauth2/authorization/google");
    }

    @GetMapping("/google/redirect")
    public ResponseEntity<?> handleRedirect() {
        return ResponseEntity.ok("Redirected from Google. Processing...");
    }

    @GetMapping("/google/success")
    public ResponseEntity<?> loginSuccess(OAuth2AuthenticationToken authenticationToken) {
        OAuth2User oAuth2User = "**********"
        String email = oAuth2User.getAttribute("email");
        String username = oAuth2User.getAttribute("name");
        String profileImage = oAuth2User.getAttribute("picture");
        String oauthProvider = "**********"
        String oauthId = oAuth2User.getName();

        OauthSignupRequestDto dto = new OauthSignupRequestDto();
        dto.setEmail(email);
        dto.setUsername(username);
        dto.setProfileImage(profileImage);
        dto.setOauthProvider(oauthProvider);
        dto.setOauthId(oauthId);

        authService.registerOauthUser(dto);

        return ResponseEntity.ok().body(authService.oauthLogin(email, oauthProvider, oauthId));
    }

    @GetMapping("/google/failure")
    public ResponseEntity<?> loginFail() {
        return ResponseEntity.status(401).body("Login failed. Please try again.");
    }
}
ntity.status(401).body("Login failed. Please try again.");
    }
}
