#date: 2024-09-20T17:01:51Z
#url: https://api.github.com/gists/ead16838a0474577d77f17a0e7843cdf
#owner: https://api.github.com/users/Zolyn

import os
import click
import json
from base64 import b64decode
# https://github.com/wqhanginge/pgmm_decrypt/tree/bugfix/incorrect-decryption
from pgmm_decrypt import decrypt_pgmm_key, decrypt_pgmm_resource

@click.group()
def main():
    """\b
    ╔═╗╔═╗╔╦╗╔╦╗  ╔═╗┌─┐┌┬┐┌─┐┌─┐
    ╠═╝║ ╦║║║║║║  ║  │ │ ││├┤ │  
    ╩  ╚═╝╩ ╩╩ ╩  ╚═╝└─┘─┴┘└─┘└─┘
    Pixel Game Maker MV Codec @syrinka

    pgmm-codec COMMAND --help 查看对应命令的帮助

    \b
    Thanks to:
    https://github.com/blluv/pgmm_decrypt
    """
    pass

@main.command('decrypt', short_help='使用密钥解密文件')
@click.argument('info', type=click.Path(exists=True, file_okay=True, readable=True))
@click.option('-i', '--input',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='''\b
    输入目录，将会尝试解密其下的*所有文件*
    请留意当中是否仍有正常文件，解密后很可能打不开''')
@click.option('-o', '--output',
    type=click.Path(file_okay=False),
    help='输出目录，留空则为 <输入目录>-dec')
@click.option('-w', '--weak',
    is_flag=True,
    help='使用弱解密模式')
@click.option('-q', '--quiet',
    is_flag=True)
def func(info, input, output, weak, quiet):
    """
    使用密钥解密资源，并输出到指定路径

    INFO: info.json文件

    \b
    使用例：
        pgmm-codec decrypt INFO -i Resources/img -o img-dec
    """
    with open(info, "r", encoding="utf-8") as f:
        encrypted_key = b64decode(json.load(f)["key"])
    decrypted_key = decrypt_pgmm_key(encrypted_key)

    if output is None:
        output = input + '-dec'

    ilen = len(input)
    for root, dirs, files in os.walk(input):
        if not quiet:
            print(f'# 当前目录：{root}')

        for file in files:
            ipath = os.path.join(root, file)
            opath = os.path.join(output, root[ilen+1:], file)

            with open(ipath, "rb") as f:
                file_bytes = f.read()
            
            decrypted_bytes = decrypt_pgmm_resource(file_bytes, decrypted_key, weak=weak)
            
            os.makedirs(os.path.dirname(opath), exist_ok=True)

            with open(opath, "wb") as f:
                f.write(decrypted_bytes)

            if not quiet:
                print(file)


if __name__ == '__main__':
    main()
