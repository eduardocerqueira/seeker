//date: 2022-07-14T17:13:02Z
//url: https://api.github.com/gists/92bd2f5194a7640b6343aed5cbf3de96
//owner: https://api.github.com/users/shaozc3g

@PostMapping("/Content")
//@Async(value = ExecuteName.MEDIA)
public DataResult mediaAudit(@RequestParam(value = "Thumbnail", required = false) MultipartFile thumbnail,
                             @RequestParam("File") MultipartFile file,
                             HttpServletRequest request,
                             HttpServletResponse response)  {
    response.setHeader("Content-Type", "application/json;charset=UTF-8");
    response.setHeader("tid", UUID.randomUUID().toString());
    response.setStatus(HttpServletResponse.SC_OK);
    Boolean result = mediaFrontService.mediaAudit(thumbnail, file, request);
    if (result) {
        return DataResult.success();
    }
    return DataResult.getDataResult(MediaServerResponseCode.MEDIA_BUSY);
}


//=========================================

@Override
public Boolean mediaAudit(MultipartFile thumbnail, MultipartFile file) {
    try {
        // 缩略图处理
        if (!thumbnail.isEmpty()) {
            // 获取原始文件名称
            String thumbnailFilename = thumbnail.getOriginalFilename();
            // 保存文件
            thumbnail.transferTo(new File("/Users/rambo/Desktop/Temp/" + thumbnailFilename));
        }

        // 正文图处理
        if (!file.isEmpty()) {
            // 获取原始文件名称
            String filename = file.getOriginalFilename();
            // 保存文件
            file.transferTo(new File("/Users/rambo/Desktop/Temp/" + filename));
        } else {
            throw new BusinessException(MediaServerResponseCode.MEDIA_PARAMS);
        }
    } catch (IOException e) {
        e.printStackTrace();
        throw new BusinessException(MediaServerResponseCode.MEDIA_BUSY.getCode(), e.getMessage());
    }
    return true;
}