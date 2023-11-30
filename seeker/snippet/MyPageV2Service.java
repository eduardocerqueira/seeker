//date: 2023-11-30T17:08:19Z
//url: https://api.github.com/gists/42cf15f7e82f395a729602c02f69b521
//owner: https://api.github.com/users/chaeeun-Han

package com.hamahama.pupmory.service;

import com.hamahama.pupmory.domain.memorial.Comment;
import com.hamahama.pupmory.domain.memorial.CommentRepositoryImpl;
import com.hamahama.pupmory.domain.memorial.Post;
import com.hamahama.pupmory.domain.memorial.PostRepository;
import com.hamahama.pupmory.domain.mypage.Announcement;
import com.hamahama.pupmory.domain.mypage.AnnouncementRepository;
import com.hamahama.pupmory.domain.user.ServiceUser;
import com.hamahama.pupmory.domain.user.ServiceUserRepository;
import com.hamahama.pupmory.dto.mypage.AnnouncementDetailDto;
import com.hamahama.pupmory.dto.mypage.AnnouncementMetaDto;
import com.hamahama.pupmory.dto.mypage.CommentMetaDto;
import com.hamahama.pupmory.pojo.ErrorMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * @author Queue-ri
 * @since 2023/11/18
 */

@RequiredArgsConstructor
@Service
@Slf4j
public class MyPageV2Service {
    private final AnnouncementRepository anRepo;
    private final CommentRepositoryImpl commentRepo;
    private final PostRepository postRepo;
    private final ServiceUserRepository userRepo;

    @Transactional
    public List<AnnouncementMetaDto> getAllAnnouncementMeta() {
        List<Announcement> anList = anRepo.findAll();
        List<AnnouncementMetaDto> dtoList = new ArrayList<AnnouncementMetaDto>();

        for (Announcement an : anList)
            dtoList.add(AnnouncementMetaDto.of(an));

        return dtoList;
    }

    @Transactional
    public ResponseEntity<?> getAnnouncementDetail(Long aid) {
        Optional<Announcement> optAnnouncement = anRepo.findById(aid);

        if (optAnnouncement.isPresent()) {
            AnnouncementDetailDto dto = AnnouncementDetailDto.of(optAnnouncement.get());
            return new ResponseEntity<AnnouncementDetailDto>(dto, HttpStatus.OK);
        } else
            return new ResponseEntity<ErrorMessage>(new ErrorMessage(404, "no data found."), HttpStatus.NOT_FOUND);
    }

    @Transactional
    public ResponseEntity<?> getAllCommentMeta(String uid) {
        Optional<ServiceUser> optUser = userRepo.findById(uid);
        if (optUser.isEmpty())
            return new ResponseEntity<ErrorMessage>(new ErrorMessage(401, "Could not find the user."), HttpStatus.UNAUTHORIZED);

        log.info("=== start: findAllByUser ===");
        List<Comment> commentList = commentRepo.findAllByUser(optUser.get());
        log.info("=== end: findAllByUser ===");
        List<CommentMetaDto> dtoList = new ArrayList<>();

        for (Comment comment : commentList) {
            log.info("=== start: postRepo.findById ===");
            Post post = postRepo.findById(comment.getPost().getId()).get();
            log.info("=== end: postRepo.findById ===");

            log.info("=== start: userRepo.findById ===");
            ServiceUser user = userRepo.findById(post.getUserUid()).get();
            log.info("=== end: userRepo.findById ===");
            dtoList.add(CommentMetaDto.of(comment, user.getNickname()));
        }

        return new ResponseEntity<List<CommentMetaDto>>(dtoList, HttpStatus.OK);
    }

}
