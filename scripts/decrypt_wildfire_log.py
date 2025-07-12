# -*- coding: utf-8 -*-

# usage
# python3 decrypt_wildfire_log.py [input.wfire] [output.json]

# see downloadEventLog function:
# https://github.com/iann0036/wildfire/blob/9bea9b4adb3399cba6020dcada760d329af9219f/app.js
# code from:
# https://stackoverflow.com/questions/36762098/how-to-decrypt-password-from-javascript-cryptojs-aes-encryptpassword-passphras

import sys

import argparse
from pathlib import Path

import json
from Cryptodome import Random
from Cryptodome.Cipher import AES
import base64
from hashlib import md5
import unicodedata

BLOCK_SIZE = 16


def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
    return str(text)

def pad(data):
    length = BLOCK_SIZE - (len(data) % BLOCK_SIZE)
    return data + (chr(length)*length).encode()

def unpad(data):
    return data[:-(data[-1] if type(data[-1]) == int else ord(data[-1]))]

def bytes_to_key(data, salt, output=48):
    # extended from https://gist.github.com/gsakkis/4546068
    assert len(salt) == 8, len(salt)
    data += salt
    key = md5(data).digest()
    final_key = key
    while len(final_key) < output:
        key = md5(key + data).digest()
        final_key += key
    return final_key[:output]

def encrypt(message, passphrase):
    salt = Random.new().read(8)
    key_iv = bytes_to_key(passphrase, salt, 32+16)
    key = key_iv[:32]
    iv = key_iv[32:]
    aes = AES.new(key, AES.MODE_CBC, iv)
    return base64.b64encode(b"Salted__" + salt + aes.encrypt(pad(message)))

def decrypt(encrypted, passphrase):
    encrypted = base64.b64decode(encrypted)
    assert encrypted[0:8] == b"Salted__"
    salt = encrypted[8:16]
    key_iv = bytes_to_key(passphrase, salt, 32+16)
    key = key_iv[:32]
    iv = key_iv[32:]
    aes = AES.new(key, AES.MODE_CBC, iv)
    return unpad(aes.decrypt(encrypted[16:]))


def main():
    if len(sys.argv) < 3:
        raise Exception('Inform an input file (e.g. WildfireExport_1607695706.txt) and an output file (e.g. output.json)')

    # read encrypted file
    with open(sys.argv[1], encoding='utf8') as f:
        ct_b64 = f.read()

    # decrypt steps
    password = "3ur9480tvb439f83r8".encode()
    pt = decrypt(ct_b64, password)
    raw_pt = str(pt)[2:-1]

    # save decrypted file
    try:
        dict_pt = json.loads(raw_pt)
    except:
        raw_pt = raw_pt.encode('utf-8').decode('unicode_escape')
        dict_pt = json.loads(raw_pt)

    with open(sys.argv[2], 'w', encoding='utf8') as f:
        json.dump(dict_pt, f, ensure_ascii=False, indent=4)

    return dict_pt


if __name__ == '__main__':
    main()
