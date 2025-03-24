//date: 2025-03-24T17:04:26Z
//url: https://api.github.com/gists/899dabd9c7082254a1606848f5a014c8
//owner: https://api.github.com/users/DarkRubin

if (item.isDir()) {
                resourceResponseDtos.add(ResourceResponseDto.builder()
                        .name(resourceNamingUtil.getFileNameWithoutPath(item.objectName()))
                        .path(path)
                        .size(item.size())
                        .type("DIRECTORY")
                        .build());
            } else {
                resourceResponseDtos.add(ResourceResponseDto.builder()
                        .name(resourceNamingUtil.getFileNameWithoutPath(item.objectName()))
                        .path(path)
                        .size(item.size())
                        .type("FILE")
                        .build());
            }