import re, sys, zlib

ESC = {ord('n'): 10, ord('r'): 13, ord('t'): 9, ord('b'): 8, ord('f'): 12}


def unescape(s):
    out, i, n = bytearray(), 0, len(s)
    while i < n:
        c = s[i]
        if c != 0x5C:
            out.append(c); i += 1; continue
        i += 1
        if i >= n: break
        e = s[i]
        if e in ESC:
            out.append(ESC[e]); i += 1
        elif e in (0x28, 0x29, 0x5C):
            out.append(e); i += 1
        elif 0x30 <= e <= 0x37:
            d = bytearray()
            while i < n and len(d) < 3 and 0x30 <= s[i] <= 0x37:
                d.append(s[i]); i += 1
            out.append(int(d, 8) & 0xFF)
        elif e in (0x0A, 0x0D):
            i += 1
        else:
            out.append(e); i += 1
    return bytes(out)


def lit(content, i):
    i += 1; depth = 1; buf = bytearray(); n = len(content)
    while i < n:
        c = content[i]
        if c == 0x5C:
            buf.append(c); i += 1
            if i < n:
                buf.append(content[i]); i += 1
            continue
        if c == 0x28:
            depth += 1
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return unescape(bytes(buf)), i + 1
        buf.append(c); i += 1
    return unescape(bytes(buf)), i


def hexi(content, i):
    j = content.find(b'>', i)
    if j == -1:
        return b'', len(content)
    hx = re.sub(rb'[^0-9A-Fa-f]', b'', content[i + 1:j])
    if len(hx) % 2:
        hx += b'0'
    try:
        return bytes.fromhex(hx.decode()), j + 1
    except ValueError:
        return b'', j + 1


def text_of(content):
    out, i, n = [], 0, len(content)
    while i < n:
        c = content[i]
        if c == 0x28:
            s, i = lit(content, i); out.append(s); continue
        if c == 0x3C and i + 1 < n and content[i + 1] != 0x3C:
            s, i = hexi(content, i); out.append(s); continue
        if content.startswith((b'ET', b'T*', b'Td', b'TD'), i):
            out.append(b'\n'); i += 2; continue
        if content.startswith((b'TJ', b'Tj'), i):
            i += 2; continue
        i += 1
    return b''.join(out)


def streams(data):
    got = []
    for m in re.finditer(rb'stream\r?\n', data):
        start = m.end()
        end = data.find(b'endstream', start)
        if end == -1:
            continue
        raw = data[start:end].rstrip(b'\r\n')
        try:
            got.append(zlib.decompress(raw))
        except zlib.error:
            try:
                got.append(zlib.decompressobj().decompress(raw))
            except zlib.error:
                pass
    return got


def main(path):
    data = open(path, 'rb').read()
    chunks = [c for c in streams(data) if b'Tj' in c or b'TJ' in c]
    sys.stderr.write('%d text streams\n' % len(chunks))
    t = text_of(b'\n'.join(chunks)).decode('latin-1')
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r' *\n *', '\n', t)
    t = re.sub(r'\n{2,}', '\n', t)
    return t.strip()


if __name__ == '__main__':
    open(sys.argv[2], 'w', encoding='utf-8').write(main(sys.argv[1]))
