//date: 2023-11-30T17:08:19Z
//url: https://api.github.com/gists/42cf15f7e82f395a729602c02f69b521
//owner: https://api.github.com/users/chaeeun-Han

package com.hamahama.pupmory.controller;

import com.hamahama.pupmory.dto.mypage.AnnouncementMetaDto;
import com.hamahama.pupmory.service.MyPageV2Service;
import com.hamahama.pupmory.util.auth.JwtKit;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * @author Queue-ri
 * @since 2023/11/18
 */

@RequiredArgsConstructor
@RestController
@RequestMapping("v2/mypage")
public class MyPageV2Controller {
    private final MyPageV2Service myPageService;
    private final JwtKit jwtKit;

    @GetMapping("/announcement/all")
    public List<AnnouncementMetaDto> getAllAnnouncementMeta() {
        return myPageService.getAllAnnouncementMeta();
    }

    @GetMapping("/announcement")
    public ResponseEntity<?> getAnnouncementDetail(@RequestParam Long aid) {
        return myPageService.getAnnouncementDetail(aid);
    }

    @GetMapping("/comment/all")
    public ResponseEntity<?> getAllCommentMeta() {
        String uid = "axNNnzcfJaSiTPI6kW23G2Vns9o1";
        return myPageService.getAllCommentMeta(uid);
    }
}
