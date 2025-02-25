#date: 2025-02-25T16:52:12Z
#url: https://api.github.com/gists/e4cf54c52f31af4171c479d6cad153df
#owner: https://api.github.com/users/yodamaster

FROM scratch
COPY rootfs/ /
COPY busybox /bin
RUN ["./bin/busybox", "--install", "-s", "/bin"]
CMD ["/bin/sh"]