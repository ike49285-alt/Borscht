//! A minimal PNG writer.
//!
//! Deliberately dependency-free: the whole project pulls in nothing but rayon,
//! and an image encoder is not worth breaking that for. Pixel data goes into
//! stored (uncompressed) deflate blocks, which costs file size but needs no
//! compressor. Frames are intermediate artefacts, not deliverables.

fn crc32(data: &[u8]) -> u32 {
    // Table built on the fly; this runs a handful of times per frame.
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *slot = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    let mut body = Vec::with_capacity(4 + payload.len());
    body.extend_from_slice(kind);
    body.extend_from_slice(payload);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

/// Encode 8-bit RGB. `pixels` must be `width * height * 3` bytes.
pub fn encode_rgb(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    assert_eq!(pixels.len(), (width as usize) * (height as usize) * 3);

    // PNG requires a filter byte at the start of every scanline.
    let mut raw = Vec::with_capacity(pixels.len() + height as usize);
    for y in 0..height as usize {
        raw.push(0); // filter: none
        let row = y * width as usize * 3;
        raw.extend_from_slice(&pixels[row..row + width as usize * 3]);
    }

    // zlib wrapper around stored deflate blocks.
    let mut z = vec![0x78, 0x01];
    let mut offset = 0usize;
    while offset < raw.len() {
        let len = (raw.len() - offset).min(65535);
        let last = offset + len >= raw.len();
        z.push(if last { 1 } else { 0 });
        z.extend_from_slice(&(len as u16).to_le_bytes());
        z.extend_from_slice(&(!(len as u16)).to_le_bytes());
        z.extend_from_slice(&raw[offset..offset + len]);
        offset += len;
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = Vec::with_capacity(z.len() + 128);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, truecolour RGB
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_a_valid_png_container() {
        let px = vec![128u8; 4 * 3 * 3];
        let png = encode_rgb(4, 3, &px);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR is always a 13-byte payload.
        assert_eq!(&png[8..12], &13u32.to_be_bytes());
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..20], &4u32.to_be_bytes());
        assert_eq!(&png[20..24], &3u32.to_be_bytes());
        assert!(png.ends_with(&[0xAE, 0x42, 0x60, 0x82]), "missing IEND crc");
    }

    #[test]
    fn crc_matches_known_values() {
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn adler_matches_known_values() {
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"abc"), 0x024D_0127);
    }

    #[test]
    fn large_images_span_multiple_stored_blocks() {
        // Forces more than one 65535-byte deflate block.
        let px = vec![7u8; 400 * 400 * 3];
        let png = encode_rgb(400, 400, &px);
        assert!(png.len() > 65_535 * 2);
        assert!(png.ends_with(&[0xAE, 0x42, 0x60, 0x82]));
    }
}
