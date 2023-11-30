//date: 2023-11-30T17:08:19Z
//url: https://api.github.com/gists/42cf15f7e82f395a729602c02f69b521
//owner: https://api.github.com/users/chaeeun-Han

package com.hamahama.pupmory.controller;

import com.hamahama.pupmory.dto.mypage.AnnouncementMetaDto;
import com.hamahama.pupmory.service.MyPageService;
import com.hamahama.pupmory.util.auth.JwtKit;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * @author Queue-ri
 * @since 2023/11/18
 */

@RequiredArgsConstructor
@RestController
@RequestMapping("mypage")
public class MyPageController {
    private final MyPageService myPageService;
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
    public ResponseEntity<?> getAllCommentMeta(@RequestHeader(value = "**********"
        String uid = "**********"
        return myPageService.getAllCommentMeta(uid);
    }
}
etAllCommentMeta(uid);
    }
}
