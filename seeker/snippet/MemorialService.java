//date: 2023-11-30T17:08:19Z
//url: https://api.github.com/gists/42cf15f7e82f395a729602c02f69b521
//owner: https://api.github.com/users/chaeeun-Han

package com.hamahama.pupmory.service;

import com.hamahama.pupmory.domain.memorial.*;
import com.hamahama.pupmory.domain.user.ServiceUser;
import com.hamahama.pupmory.domain.user.ServiceUserRepository;
import com.hamahama.pupmory.dto.memorial.*;
import com.hamahama.pupmory.pojo.ErrorMessage;
import com.hamahama.pupmory.pojo.PostMeta;
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
import java.util.Optional;

/**
 * @author Queue-ri
 * @since 2023/09/11
 */

@Slf4j
@RequiredArgsConstructor
@Service
public class MemorialService {
    private final PostRepository postRepo;
    private final UserLikeRepository likeRepo;
    private final CommentRepository commentRepo;
    private final ServiceUserRepository userRepo;
    private final S3Uploader s3Uploader;

    @Transactional
    public PostDetailResponseDto getPost(Long id) {
        Post post = postRepo.findById(id).get();
        Long likeCount = likeRepo.countByPostId(id);
        return PostDetailResponseDto.of(post, likeCount);
    }

    @Transactional
    public ResponseEntity<?> deletePost(String uid, Long postId) {
        Optional<Post> optPost = postRepo.findById(postId);

        if (optPost.isPresent()) {
            // check if requester == OP
            String opUid = optPost.get().getUserUid();

            if (opUid.equals(uid)) {
                commentRepo.deleteAllByPostId(postId); // 해당 글의 댓글 전부 삭제
                postRepo.deleteById(postId);
                return new ResponseEntity<>(HttpStatus.OK);
            } else {
                return new ResponseEntity<ErrorMessage>(new ErrorMessage(403, "You do not have permission to delete this post."), HttpStatus.FORBIDDEN);
            }
        } else {
            return new ResponseEntity<ErrorMessage>(new ErrorMessage(404, "Could not find the post."), HttpStatus.NOT_FOUND);
        }
    }

    @Transactional
    public PostAllResponseDto getAllPost(String uid) {
        List<Post> posts = postRepo.findAllByUserUid(uid);
        List<PostMeta> postMetas = new ArrayList<>();
        for (Post post : posts)
            postMetas.add(PostMeta.of(post));

        ServiceUser user = userRepo.findByUserUid(uid);

        return new PostAllResponseDto(user.getNickname(), user.getProfileImage(), user.getPuppyName(), user.getPuppyType(), user.getPuppyAge(), postMetas);
    }

    @Transactional
    public PostAllResponseDto getOthersAllPost(String targetUid) {
        // myUid가 targetUid의 메모리얼을 조회: private은 조회되지 않음
        List<Post> posts = postRepo.findAllByUserUidAndIsPrivateFalse(targetUid);
        List<PostMeta> postMetas = new ArrayList<>();
        for (Post post : posts)
            postMetas.add(PostMeta.of(post));

        ServiceUser user = userRepo.findByUserUid(targetUid);

        return new PostAllResponseDto(user.getNickname(), user.getProfileImage(), user.getPuppyName(), user.getPuppyType(), user.getPuppyAge(), postMetas);
    }

    @Transactional
    public List<FeedPostResponseDto> getFeedByLatest(String uuid) {
        List<Post> posts = postRepo.findLatestFeed(uuid);
        List<FeedPostResponseDto> feeds = new ArrayList<>();

        for (Post post : posts) {
            ServiceUser user = userRepo.findByUserUid(post.getUserUid());
            FeedPostResponseDto dto = FeedPostResponseDto.of(post, user);
            feeds.add(dto);
        }

        return feeds;
    }

    @Transactional
    public List<FeedPostResponseDto> getFeedByFilter(String uuid, FeedPostFilterRequestDto fDto) {
        List<Post> posts = new ArrayList<>();
        if (fDto.getType() == null) posts = postRepo.findFilteredFeedByAge(uuid, fDto.getAge());
        else if (fDto.getAge() == null) posts = postRepo.findFilteredFeedByType(uuid, fDto.getType());
        else posts = postRepo.findFilteredFeedByBoth(uuid, fDto.getType(), fDto.getAge());

        List<FeedPostResponseDto> feeds = new ArrayList<>();

        for (Post post : posts) {
            ServiceUser user = userRepo.findByUserUid(post.getUserUid());
            FeedPostResponseDto dto = FeedPostResponseDto.of(post, user);
            feeds.add(dto);
        }

        return feeds;
    }

    @Transactional
    public void savePost(String uid, PostRequestDto dto, List<MultipartFile> mfiles) throws IOException {
        if (mfiles != null)
            for (MultipartFile mfile : mfiles)
                log.info("- image: " + mfile);

        List<String> fileUrlList = new ArrayList<String>();
        if (mfiles != null)
            fileUrlList = s3Uploader.upload(mfiles, "memorial", uid);

        postRepo.save(dto.toEntity(uid, fileUrlList));
    }

    @Transactional
    public boolean getLike(String uid, Long postId) {
        return likeRepo.findByUserUidAndPostId(uid, postId).isPresent();
    }

    @Transactional
    public void processLike(String uid, Long postId) {
        Optional<UserLike> like = likeRepo.findByUserUidAndPostId(uid, postId);

        if (like.isPresent()) { // 이미 좋아요 한 상태
            likeRepo.deleteByUserUidAndPostId(uid, postId);
        } else { // 좋아요 안한 상태
            likeRepo.save(
                    UserLike.builder()
                            .userUid(uid)
                            .postId(postId)
                            .build()
            );
        }
    }

    @Transactional
    public List<CommentResponseDto> getComment(Long postId) {
        List<Comment> commentList = commentRepo.findAllByPostId(postId);

        List<CommentResponseDto> dtoList = new ArrayList<CommentResponseDto>();
        for (Comment comment : commentList) {
            ServiceUser user = userRepo.findById(comment.getUser().getUserUid()).get();
            dtoList.add(CommentResponseDto.of(comment, user));
        }

        return dtoList;
    }
    
    @Transactional
    public ResponseEntity<?> saveComment(String uid, Long postId, CommentRequestDto dto) {
        Optional<ServiceUser> optUser = userRepo.findById(uid);
        if (optUser.isPresent()) {
            commentRepo.save(dto.toEntity(optUser.get()));
        } else {
            return new ResponseEntity<ErrorMessage>(new ErrorMessage(401, "Could not find the user."), HttpStatus.NOT_FOUND);
        }
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @Transactional
    public ResponseEntity<?> deleteComment(String uid, Long cid) {
        Optional<Comment> optComment = commentRepo.findById(cid);

        if (optComment.isPresent()) {
            // check if requester == OP
            String opUid = optComment.get().getUser().getUserUid();

            if (opUid.equals(uid)) {
                commentRepo.deleteById(cid);
                return new ResponseEntity<>(HttpStatus.OK);
            } else {
                return new ResponseEntity<ErrorMessage>(new ErrorMessage(403, "You do not have permission to delete this comment."), HttpStatus.FORBIDDEN);
            }
        } else {
            return new ResponseEntity<ErrorMessage>(new ErrorMessage(404, "Could not find the comment."), HttpStatus.NOT_FOUND);
        }
    }
}
