//date: 2021-11-24T17:03:37Z
//url: https://api.github.com/gists/fdfd501775dad79b0e3b8d58dbe4a685
//owner: https://api.github.com/users/flavio

#!/bin/bash

INPUT_NAMES=(
  "busybox"
  "test.com:tag"
  "test.com:5000"
  "test.com/repo:tag"
  "test:5000/repo:tag"
  "test:5000/repo@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
  "test:5000/repo:tag@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
  "lowercase:Uppercase"
  "sub-dom1.foo.com/bar/baz/quux"
  "sub-dom1.foo.com/bar/baz/quux:some-long-tag"
  "b.gcr.io/test.example.com/my-app:test.example.com"
  "xn--n3h.com/myimage:xn--n3h.com"
  "xn--7o8h.com/myimage:xn--7o8h.com@sha512:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
  "foo_bar.com:8080"
  "foo/foo_bar.com:8080"
)

for image in "${INPUT_NAMES[@]}"
do
  echo "$image -> $(./container-image-name ${image})"
done