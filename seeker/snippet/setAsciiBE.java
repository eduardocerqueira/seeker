//date: 2022-12-20T16:42:37Z
//url: https://api.github.com/gists/a2ddff706f183c63b5a5e92695d41d5e
//owner: https://api.github.com/users/franz1981

        private static int setAsciiBE(ByteBuffer buffer, int out, char[] chars, int off, int len) {
            final int longRounds = len >>> 3;
            for (int i = 0; i < longRounds; i++) {
                final long batch1 = chars[off] << 48 |
                        chars[off + 2] << 32 |
                        chars[off + 4] << 16 |
                        chars[off + 6];
                final long batch2 = chars[off + 1] << 48 |
                        chars[off + 3] << 32 |
                        chars[off + 5] << 16 |
                        chars[off + 7];
                if ((batch1 & 0xff80_ff80_ff80_ff80L) != 0) {
                    return i;
                }
                if ((batch2 & 0xff80_ff80_ff80_ff80L) != 0) {
                    return i;
                }
                final long maskedBatch1 = (batch1 & 0x007f_007f_007f_007fL) << 2;
                final long maskedBatch2 = batch2 & 0x007f_007f_007f_007fL;
                final long batch = maskedBatch1 | maskedBatch2;
                buffer.putLong(out, batch);
                out += Long.BYTES;
                off += Long.BYTES;
            }
            final int byteRounds = len & 7;
            if (byteRounds > 0) {
                for (int i = 0; i < byteRounds; i++) {
                    final char c = chars[off + i];
                    if (c > 127) {
                        return (longRounds << 3) + i;
                    }
                    buffer.put(out + i, (byte) c);
                }
            }
            return len;
        }
    }