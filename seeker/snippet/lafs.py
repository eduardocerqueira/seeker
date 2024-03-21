#date: 2024-03-21T16:54:59Z
#url: https://api.github.com/gists/d8cff42e08bfbd9a7dcaf8b43b543977
#owner: https://api.github.com/users/xmba15

@dataclass
class LAFFeatures:
    lafs: torch.Tensor  # 1 x num kpts x 2 x 3
    resps: torch.Tensor  # 1 x num kpts
    descs: torch.Tensor  # 1 x num kpts x 128

    def __len__(self):
        return self.lafs.shape[1]

    @property
    def cv2_kpts(
        self,
    ) -> List[cv2.KeyPoint]:
        mkpts = kornia.feature.get_laf_center(self.lafs).squeeze()  # num kpts x 2
        scales = kornia.feature.get_laf_scale(self.lafs).flatten()  # num kpts
        orientations = kornia.feature.get_laf_orientation(self.lafs).flatten()  # num kpts
        responses = self.resps.flatten()  # num kpts

        cv2_kpts = []
        for (x, y), scale, orientation, response in zip(mkpts, scales, orientations, responses):
            cv2_kpts.append(
                cv2.KeyPoint(
                    x=x,
                    y=y,
                    _size=scale,
                    _angle=orientation,
                    _response=response,
                )
            )

        return cv2_kpts

