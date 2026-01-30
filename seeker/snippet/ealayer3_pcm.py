#date: 2026-01-30T17:29:31Z
#url: https://api.github.com/gists/8adf3ee89bdab85682df5520e5141c58
#owner: https://api.github.com/users/webmaven

import struct
import math
import argparse
import wave
import os

class EALayer3PCM:
    def __init__(self):
        self.version = 0x00
        self.codec = 0x06
        self.type = 0x01 # Streamed

    def pack_snr_header(self, channels, sample_rate, num_samples):
        # Header 1: Version(4), Codec(4), Config(6), Rate(18)
        header1 = (self.version << 28) | (self.codec << 24) | ((channels - 1) << 18) | (sample_rate & 0x3FFFF)
        # Header 2: Type(2), LoopFlag(1), Samples(29)
        header2 = (self.type << 30) | (0 << 29) | (num_samples & 0x1FFFFFFF)
        return struct.pack('>II', header1, header2)

    def unpack_snr_header(self, data):
        header1, header2 = struct.unpack('>II', data[:8])
        version = (header1 >> 28) & 0x0F
        codec = (header1 >> 24) & 0x0F
        channels = ((header1 >> 18) & 0x3F) + 1
        sample_rate = header1 & 0x3FFFF
        type_flag = (header2 >> 30) & 0x03
        num_samples = header2 & 0x1FFFFFFF
        return channels, sample_rate, num_samples

    def pack_l32p_granule(self, pcm_data, channels, sample_rate_index, channel_mode):
        # pcm_data should be a list of shorts or a bytes object
        if isinstance(pcm_data, bytes):
            pcm_bytes = pcm_data
            num_frames = len(pcm_bytes) // (channels * 2)
        else:
            pcm_bytes = struct.pack('<' + 'h' * len(pcm_data), *pcm_data)
            num_frames = len(pcm_data) // channels

        # Headers
        # Base: Extended(1), Unknown(1), Unused(2), Size(12)
        # Extended Header is used
        header_size = 2 + 4 + 2 # Base(2) + Extended(4) + MPEG(2, padded)
        total_size = header_size + len(pcm_bytes)
        
        base_header = (1 << 15) | (0 << 14) | (0 << 12) | (total_size & 0xFFF)
        
        # Extended: Mode(2), Discard(10), PCM_Count(10), Granule_Size(10)
        # Mode 0, Discard 0, MPEG size 0
        ext_header = (0 << 30) | (0 << 20) | (num_frames << 10) | 0
        
        # MPEG Params: Version(2), RateIdx(2), Mode(2), ModeExt(2), GranIdx(1)
        # MPEG 1.0 = version 3
        # Granule 0
        mpeg_params = (3 << 7) | (sample_rate_index << 5) | (channel_mode << 3) | (0 << 1) | 0
        # Padded to 16 bits
        mpeg_bytes = struct.pack('>H', mpeg_params << 7) 

        return struct.pack('>HI', base_header, ext_header) + mpeg_bytes + pcm_bytes

    def unpack_l32p_granule(self, data, channels):
        base_header, = struct.unpack('>H', data[:2])
        extended = (base_header >> 15) & 1
        total_size = base_header & 0xFFF
        
        offset = 2
        pcm_frames = 0
        if extended:
            ext_header, = struct.unpack('>I', data[offset:offset+4])
            pcm_frames = (ext_header >> 10) & 0x3FF
            offset += 4
            
        # MPEG params
        offset += 2
        
        pcm_bytes = data[offset:total_size]
        return pcm_bytes, total_size

    def encode(self, pcm_data, channels, sample_rate):
        granule_frames = 576
        num_samples = len(pcm_data) // channels
        
        snr_header = self.pack_snr_header(channels, sample_rate, num_samples)
        
        rates = [44100, 48000, 32000, 22050, 24000, 16000, 11025, 12000, 8000]
        try:
            sample_rate_index = rates.index(sample_rate)
        except ValueError:
            sample_rate_index = 0
            
        channel_mode = 0 if channels > 1 else 3
        
        sns_data = b''
        for i in range(0, num_samples, granule_frames):
            frame_count = min(granule_frames, num_samples - i)
            granule_pcm = pcm_data[i*channels : (i+frame_count)*channels]
            sns_data += self.pack_l32p_granule(granule_pcm, channels, sample_rate_index, channel_mode)
            
        block_header = struct.pack('>I', (0x00 << 24) | (len(sns_data) + 8))
        block_samples = struct.pack('>I', num_samples)
        sns_block = block_header + block_samples + sns_data
        
        sns_block += struct.pack('>I', (0x80 << 24) | 8) + struct.pack('>I', num_samples)
        
        return snr_header, sns_block

    def decode(self, snr_data, sns_data):
        channels, sample_rate, total_samples = self.unpack_snr_header(snr_data)
        
        all_pcm = []
        offset = 0
        while offset < len(sns_data):
            if offset + 4 > len(sns_data): break
            block_info, = struct.unpack('>I', sns_data[offset:offset+4])
            block_id = (block_info >> 24) & 0xFF
            block_size = block_info & 0x00FFFFFF
            
            if block_id == 0x80: break
                
            if block_id == 0x00:
                block_offset = offset + 8
                block_end = offset + block_size
                while block_offset < block_end:
                    pcm_bytes, granule_size = self.unpack_l32p_granule(sns_data[block_offset:], channels)
                    num_shorts = len(pcm_bytes) // 2
                    all_pcm.extend(struct.unpack('<' + 'h' * num_shorts, pcm_bytes))
                    block_offset += granule_size
            
            offset += block_size
            
        return all_pcm[:total_samples * channels], channels, sample_rate

def main():
    parser = argparse.ArgumentParser(description='EALayer3 Interleaved (Version 2) PCM Codec')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-e', '--encode', action='store_true', help='Encode WAV to SNR/SNS')
    group.add_argument('-d', '--decode', action='store_true', help='Decode SNR/SNS to WAV')
    
    parser.add_argument('input', help='Input file (WAV for encoding, SNR for decoding)')
    parser.add_argument('output', nargs='?', help='Output file (SNR base name for encoding, WAV for decoding)')
    
    args = parser.parse_args()
    codec = EALayer3PCM()
    
    if args.encode:
        with wave.open(args.input, 'rb') as wav:
            params = wav.getparams()
            if params.sampwidth != 2:
                print("Error: Only 16-bit PCM WAV is supported")
                return
            pcm_bytes = wav.readframes(params.nframes)
            num_samples = len(pcm_bytes) // (params.nchannels * 2)
            pcm_data = struct.unpack('<' + 'h' * (params.nchannels * num_samples), pcm_bytes)
            
            snr, sns = codec.encode(pcm_data, params.nchannels, params.framerate)
            
            base_output = args.output if args.output else os.path.splitext(args.input)[0]
            with open(base_output + '.snr', 'wb') as f:
                f.write(snr)
            with open(base_output + '.sns', 'wb') as f:
                f.write(sns)
            print(f"Encoded to {base_output}.snr and {base_output}.sns")
            
    elif args.decode:
        snr_file = args.input
        sns_file = os.path.splitext(snr_file)[0] + '.sns'
        
        if not os.path.exists(snr_file):
            print(f"Error: SNR file {snr_file} not found")
            return
        if not os.path.exists(sns_file):
            print(f"Error: SNS file {sns_file} not found")
            return
            
        with open(snr_file, 'rb') as f:
            snr_data = f.read()
        with open(sns_file, 'rb') as f:
            sns_data = f.read()
            
        pcm_data, channels, sample_rate = codec.decode(snr_data, sns_data)
        
        output_wav = args.output if args.output else os.path.splitext(snr_file)[0] + '.wav'
        with wave.open(output_wav, 'wb') as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(struct.pack('<' + 'h' * len(pcm_data), *pcm_data))
        print(f"Decoded to {output_wav}")

if __name__ == "__main__":
    main()
