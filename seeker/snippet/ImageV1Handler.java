//date: 2023-01-23T16:52:08Z
//url: https://api.github.com/gists/414a3ff0d811a827e6b85eb4056f0f5b
//owner: https://api.github.com/users/justahmed99

package com.ahmadthesis.image.adapter.input.rest.image.v1.router;

import com.ahmadthesis.image.adapter.input.rest.common.dto.response.BaseResponse;
import com.ahmadthesis.image.adapter.input.rest.image.v1.converter.ImageConverter;
import com.ahmadthesis.image.application.usecase.ImageCommand;
import com.ahmadthesis.image.application.usecase.ImageHistoryCommand;
import com.ahmadthesis.image.domain.entity.image.ImageHistory;
import com.ahmadthesis.image.domain.valueobject.image.Activity;
import com.ahmadthesis.image.global.utils.dates.DateUtils;
import com.ahmadthesis.image.global.utils.token.TokenExtractor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.util.annotation.Nullable;

import java.io.IOException;
import java.io.InputStream;
import java.net.URLConnection;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

@Component("ImageV1Handler")
@Slf4j
public class ImageV1Handler {
    private final ImageCommand imageCommand;
    private final ImageHistoryCommand imageHistoryCommand;
    private final ImageConverter converter;
    private final DateUtils dateUtils;
    private final TokenExtractor tokenExtractor;
    @Value("${uploads.directory}")
    private String fileDir;

    @Autowired
    public ImageV1Handler(
            ImageCommand imageCommand,
            ImageHistoryCommand imageHistoryCommand,
            ImageConverter converter,
            DateUtils dateUtils,
            TokenExtractor tokenExtractor
    ) {
        this.imageCommand = imageCommand;
        this.imageHistoryCommand = imageHistoryCommand;
        this.converter = converter;
        this.dateUtils = dateUtils;
        this.tokenExtractor = "**********"
    }

    @Nullable
    public Mono<ServerResponse> healthCheck(ServerRequest request) {
        String username = "**********"
        log.info("Server health! {}", username);
        return ServerResponse.ok()
                .bodyValue(new BaseResponse<>("hello from image API!", HttpStatus.OK.value(), null));
    }

    @Nullable
    public Mono<ServerResponse> uploadImage(ServerRequest request) {
        String username = "**********"
        if (username == null) {
            return ServerResponse.badRequest().build();
        }
        log.info("Image uploading process begin");
        return converter.extract(request)
                .flatMap(saveImageRequest -> {
                    String fileName = UUID.randomUUID() + "." + saveImageRequest.getFormat();
                    String filePath = fileDir + fileName;
                    return saveImageRequest.getImage().transferTo(Paths.get(filePath))
                            .then(converter.saveImageToDomain(saveImageRequest)
                                    .flatMap(image -> {
                                        log.info("Write image data to database");
                                        image.setCreatedAt(dateUtils.now());
                                        image.setFileName(fileName);
                                        image.setOriginalImageDir(filePath);
                                        image.setMediaType(URLConnection.guessContentTypeFromName(image.getFileName()));
                                        image.setUploaderId(username);
                                        return imageCommand.save(image);
                                    })
                                    .flatMap(response -> {
                                        log.info("Write image history UPLOAD ID: {}", response.getId());
                                        ImageHistory history = new ImageHistory();
                                        history.setId(UUID.randomUUID().toString());
                                        history.setActivity(Activity.CREATE);
                                        history.setImageId(response.getId());
                                        history.setAccessorId(UUID.randomUUID().toString());
                                        history.setCreatedAt(dateUtils.now());
                                        return imageHistoryCommand.saveHistory(history)
                                                .flatMap(imageHistory -> Mono.just(response));
                                    })
                                    .flatMap(response -> {
                                        log.info("Upload image ID: {} done", response.getId());
                                        return ServerResponse.ok()
                                                .bodyValue(new BaseResponse<>("image", HttpStatus.OK.value(), converter.convertFromDomainToUploadResponse(response)));
                                    }));
                });
    }

    @Nullable
    public Mono<ServerResponse> viewImage(ServerRequest request) {
        String username = "**********"
        if (username == null) {
            return ServerResponse.badRequest().build();
        }
        log.info("View image process begin");
        return converter.extractIdRequest(request)
                .flatMap(imageCommand::getImageById)
                .publishOn(Schedulers.boundedElastic())
                .flatMap(image -> {
                    log.info("Write image history VIEW ID: {}", image.getId());
                    ImageHistory history = new ImageHistory();
                    history.setId(UUID.randomUUID().toString());
                    history.setActivity(Activity.VIEW);
                    history.setImageId(image.getId());
                    history.setAccessorId(username);
                    history.setCreatedAt(dateUtils.now());
                    return imageHistoryCommand.saveHistory(history)
                            .flatMap(imageHistory -> Mono.just(image));
                })
                .flatMap(image -> {
                    try {
                        log.info("Image found ID:" + image.getId());
                        InputStream inputStream = Files.newInputStream(Paths.get(image.getOriginalImageDir()));
                        HttpHeaders headers = new HttpHeaders();
                        headers.setContentDispositionFormData(fileDir, fileDir);
                        return ServerResponse.ok()
                                .header(HttpHeaders.CONTENT_TYPE, image.getMediaType())
                                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + image.getFileName() + "\"")
                                .bodyValue(new InputStreamResource(inputStream));
                    } catch (IOException e) {
                        log.error(e.getMessage());
                        return Mono.error(new RuntimeException(e));
                    }
                })
                .switchIfEmpty(Mono.defer(() -> {
                    log.warn("Image not found!");
                    return ServerResponse.notFound().build();
                }))
                .onErrorResume(throwable -> ServerResponse.status(500).bodyValue(throwable));
    }

    public Mono<ServerResponse> imageDetail(ServerRequest request) {
        String username = "**********"
        if (username == null) {
            return ServerResponse.badRequest().build();
        }
        log.info("View image detail process begin");
        return converter.extractIdRequest(request)
                .flatMap(imageCommand::getImageById)
                .flatMap(image -> {
                    log.info("Write image history VIEW_DETAIL ID: {}", image.getId());
                    ImageHistory history = new ImageHistory();
                    history.setId(UUID.randomUUID().toString());
                    history.setActivity(Activity.VIEW_DETAIL);
                    history.setImageId(image.getId());
                    history.setAccessorId(username);
                    history.setCreatedAt(dateUtils.now());
                    return imageHistoryCommand.saveHistory(history)
                            .flatMap(imageHistory -> Mono.just(image));
                })
                .flatMap(image -> {
                    log.info("Image found ID:" + image.getId());
                    return ServerResponse.ok()
                            .bodyValue(new BaseResponse<>("image", HttpStatus.OK.value(), converter.convertFromDomainToResponse(image)));
                })
                .switchIfEmpty(Mono.defer(() -> {
                    log.warn("Image detail not found!");
                    return ServerResponse.notFound().build();
                }))
                .onErrorResume(throwable -> ServerResponse.status(500).bodyValue(throwable));
    }
    public Mono<ServerResponse> getImageListPage(ServerRequest request) {
        return converter.getPageRequestParams(request)
                .flatMap(requestMap -> imageCommand.getImagesPage(requestMap, null))
                .flatMap(images -> ServerResponse.ok().bodyValue(new BaseResponse<>("image list", HttpStatus.OK.value(), images)));

    }

    public Mono<ServerResponse> getImageListPagePublic(ServerRequest request) {
        return converter.getPageRequestParams(request)
                .flatMap(requestMap -> imageCommand.getImagesPage(requestMap, true))
                .flatMap(images -> ServerResponse.ok().bodyValue(new BaseResponse<>("image list", HttpStatus.OK.value(), images)));

    }

}esponse.ok().bodyValue(new BaseResponse<>("image list", HttpStatus.OK.value(), images)));

    }

}