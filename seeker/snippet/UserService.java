//date: 2023-11-30T17:08:19Z
//url: https://api.github.com/gists/42cf15f7e82f395a729602c02f69b521
//owner: https://api.github.com/users/chaeeun-Han

package com.hamahama.pupmory.service;

import com.hamahama.pupmory.domain.community.HelpRepository;
import com.hamahama.pupmory.domain.community.WordCloudRepository;
import com.hamahama.pupmory.domain.memorial.CommentRepository;
import com.hamahama.pupmory.domain.memorial.PostRepository;
import com.hamahama.pupmory.domain.memorial.UserLikeRepository;
import com.hamahama.pupmory.domain.search.SearchHistoryRepository;
import com.hamahama.pupmory.domain.user.ServiceUser;
import com.hamahama.pupmory.domain.user.ServiceUserRepository;
import com.hamahama.pupmory.dto.user.ConversationStatusUpdateDto;
import com.hamahama.pupmory.dto.user.UserInfoResponseDto;
import com.hamahama.pupmory.dto.user.UserInfoUpdateDto;
import com.hamahama.pupmory.pojo.ErrorMessage;
import com.hamahama.pupmory.util.S3Uploader;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * @author Queue-ri
 * @since 2023/09/24
 */

@RequiredArgsConstructor
@Service
@Slf4j
public class UserService {
    private final ServiceUserRepository userRepo;
    private final PostRepository postRepo;
    private final CommentRepository commentRepo;
    private final HelpRepository helpRepo;
    private final UserLikeRepository ulikeRepo;
    private final SearchHistoryRepository historyRepo;
    private final WordCloudRepository wcloudRepo;
    private final CommunityService communityService;
    private final S3Uploader s3Uploader;

    @Transactional
    public UserInfoResponseDto getUserInfo(String uid) {
        Long helpCount = communityService.getHelpCount(uid);
        ServiceUser user = userRepo.findByUserUid(uid);
        return UserInfoResponseDto.of(user, helpCount);
    }

    @Transactional
    public ResponseEntity<?> setUserConversationStatus(String uid, ConversationStatusUpdateDto dto) {
        ServiceUser user = userRepo.findById(uid).get();

        if (dto.getConversationStatus() != null) {
            String currentStatus = user.getConversationStatus();

            switch (dto.getConversationStatus()) {
                // 인트로
                case "0":
                    user.setConversationStatus("1");
                    break;
                // 초기
                case "1A":
                    user.setConversationStatus(Objects.equals(currentStatus, "1B") ? "2" : "1A");
                    break;
                case "1B":
                    user.setConversationStatus(Objects.equals(currentStatus, "1A") ? "2" : "1B");
                    break;
                // 중기
                case "2A":
                    user.setConversationStatus(Objects.equals(currentStatus, "2B") ? "3" : "2A");
                    break;
                case "2B":
                    user.setConversationStatus(Objects.equals(currentStatus, "2A") ? "3" : "2B");
                    break;
                // 후기
                case "3":
                    user.setConversationStatus("4");
                    break;
                // 종결기
                case "4":
                    user.setConversationStatus("-1");
                    break; // end
                // invalid case
                default:
                    return new ResponseEntity<ErrorMessage>(new ErrorMessage(400, "Invalid status code."), HttpStatus.BAD_REQUEST);
            }
        }

        user.setNextConversationAt(dto.getNextConversationAt());

        return new ResponseEntity<>(HttpStatus.OK);
    }

    @Transactional
    public ResponseEntity<?> updateUserInfo(String uid, UserInfoUpdateDto dto, MultipartFile mfile) {
        ServiceUser user = userRepo.findById(uid).get();
        List<MultipartFile> mfileList = new ArrayList<MultipartFile>();
        List<String> fileUrlList = new ArrayList<String>();
        mfileList.add(mfile);

        try {
            if (mfile != null) {
                fileUrlList = s3Uploader.upload(mfileList, "profile", uid);
                user.setProfileImage(fileUrlList.get(0));
            }
        } catch (IOException e) {
            log.error(e.getMessage());
            return new ResponseEntity<ErrorMessage>(new ErrorMessage(502, e.getMessage()), HttpStatus.BAD_GATEWAY);
        }

        user.setNickname(dto.getNickname());
        user.setPuppyName(dto.getPuppyName());
        user.setPuppyAge(dto.getPuppyAge());
        user.setPuppyType(dto.getPuppyType());

        return new ResponseEntity<>(HttpStatus.OK);
    }

    @Transactional
    public ResponseEntity<?> deleteAccount(String uid) {
        Optional<ServiceUser> optUser = userRepo.findById(uid);
        if (optUser.isEmpty())
            return new ResponseEntity<ErrorMessage>(new ErrorMessage(401, "Could not find the user."), HttpStatus.UNAUTHORIZED);

        try {
            commentRepo.deleteAllByUser(optUser.get());
            ulikeRepo.deleteAllByUserUid(uid);
            postRepo.deleteAllByUserUid(uid);
            historyRepo.deleteAllByUserUid(uid);
            helpRepo.deleteAllByUserUid(uid);
            wcloudRepo.deleteByUserUid(uid);
            userRepo.deleteById(uid);
        } catch (Exception e) {
            log.error(e.getMessage());
            return new ResponseEntity<ErrorMessage>(new ErrorMessage(500, e.getMessage()), HttpStatus.INTERNAL_SERVER_ERROR);
        }

        return new ResponseEntity<>(HttpStatus.OK);
    }
}
