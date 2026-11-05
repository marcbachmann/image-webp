//! Encodes a PAM losslessly at q0 m0, decodes it back and compares the pixels.
use image_webp::{ColorType, EncoderParams, WebPDecoder, WebPEncoder};

fn main() {
    let file = std::env::args().nth(1).unwrap();
    let data = std::fs::read(&file).unwrap();
    let end = data.windows(7).position(|w| w == b"ENDHDR\n").unwrap() + 7;
    let text = std::str::from_utf8(&data[..end]).unwrap();
    let get = |k: &str| text.lines().find(|l| l.starts_with(k)).unwrap()[k.len()..].trim().parse::<u32>().unwrap();
    let (w, h, depth) = (get("WIDTH"), get("HEIGHT"), get("DEPTH"));
    let px = &data[end..end + (w * h * depth) as usize];
    let color = if depth == 4 { ColorType::Rgba8 } else { ColorType::Rgb8 };
    let mut out = Vec::new();
    let mut enc = WebPEncoder::new(&mut out);
    let mut params = EncoderParams::default();
    params.lossless_quality = 0;
    params.lossless_method = 0;
    params.lossless_exact = true;
    params.fast_mode = std::env::var_os("FAST").is_some();
    enc.set_params(params);
    enc.encode(px, w, h, color).unwrap();
    let mut dec = WebPDecoder::new(std::io::Cursor::new(&out)).unwrap();
    let mut back = vec![0; dec.output_buffer_size().unwrap()];
    dec.read_image(&mut back).unwrap();
    let differ = back.iter().zip(px).filter(|(a, b)| a != b).count();
    println!("{file}: {} bytes, {differ} bytes differ after decode", out.len());
    std::process::exit(i32::from(differ != 0));
}
