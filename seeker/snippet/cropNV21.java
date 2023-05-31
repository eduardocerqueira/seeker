//date: 2023-05-31T16:54:02Z
//url: https://api.github.com/gists/d6f110d264fca0bb4e0840cb63524800
//owner: https://api.github.com/users/timotismjntk

// https://www.programmersought.com/article/75461140907/

/**
   * NV21 cropping algorithm efficiency 3ms
   *
   * @param src source data
   * @param width source width
   * @param height source height
   * @param left vertex coordinates
   * @param top vertex coordinates
   * @param clip_w Cropped width
   * @param clip_h High after cropping
   * @return Cropped data
*/
    public static byte[] cropNV21(byte[] src, int width, int height, int left, int top, int clip_w, int clip_h) {
        if (left > width || top > height) {
            return null;
        }
        / / Take the couple
        int x = left * 2 / 2, y = top * 2 / 2;
        int w = clip_w * 2 / 2, h = clip_h * 2 / 2;
        int y_unit = w * h;
        int uv = y_unit / 2;
        byte[] nData = new byte[y_unit + uv];
        int uv_index_dst = w * h - y / 2 * w;
        int uv_index_src = width * height + x;
        int srcPos0 = y * width;
        int destPos0 = 0;
        int uvSrcPos0 = uv_index_src;
        int uvDestPos0 = uv_index_dst;
        for (int i = y; i < y + h; i++) {
            System.arraycopy(src, srcPos0 + x, nData, destPos0, w);//y memory block copy
            srcPos0 += width;
            destPos0 += w;
            if ((i & 1) == 0) {
                System.arraycopy(src, uvSrcPos0, nData, uvDestPos0, w);//uv memory block copy
                uvSrcPos0 += width;
                uvDestPos0 += w;
            }
        }
        return nData;
    }
