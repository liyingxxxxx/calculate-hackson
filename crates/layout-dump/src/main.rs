//! `layout-dump taesd`：把 `crates/taesd/src/layout.rs` 打成 JSON（stdout）。
//!
//! 为什么不让 Python 自己算偏移：设备程序（Rust、编译期常量）与主机打包要是各写一份，改一边忘一边就是静默的错数。
//! 这里用 `#[path]` 把设备 crate 的 layout.rs 原样编进主机程序（同 calBLAS `examples/tiny/pack` 的做法），Python 只读这份 JSON。
//! 不引第三方 crate（serde 之类）：JSON 手写，键都是固定的。
#[path = "../../taesd/src/layout.rs"]
#[allow(dead_code)]
mod taesd;

fn taesd_json() -> String {
    use taesd as L;
    let mut convs = Vec::new();
    for (i, c) in L::CONVS.iter().enumerate() {
        convs.push(format!(
            "{{\"i\": {i}, \"key\": \"{}\", \"cin\": {}, \"cout\": {}, \"cin_pad\": {}, \"cout_pad\": {}, \"bias\": {}, \"res\": {}, \"w_off\": {}, \"w_len\": {}, \"b_off\": {}, \"b_len\": {}}}",
            c.key, c.cin, c.cout, c.cin_pad, c.cout_pad, c.bias, c.res, L::W_OFF[i], L::w_len(i), L::B_OFF[i], L::b_len(i)
        ));
    }
    let maps: Vec<String> = L::MAPS.iter().map(|m| format!("[{}, {}, {}]", m[0], m[1], m[2])).collect();
    let res: Vec<String> = L::RES.iter().map(ToString::to_string).collect();
    format!(
        "{{\n \"program\": \"taesd\",\n \"consts\": {{\"B\": {}, \"LAT\": {}, \"C\": {}, \"CIN0\": {}, \"COUT\": {}, \"MARGIN_PX\": {}, \"IN_BYTES\": {}, \"OUT_BYTES\": {}, \"W_BYTES\": {}, \"ACT_BYTES\": {}, \"ZERO_BLOCKS\": {}, \"L0\": {}}},\n \"decls\": {{\"in_x\": {}, \"out_y\": {}, \"w\": {}, \"act\": {}}},\n \"res\": [{}],\n \"maps\": [{}],\n \"convs\": [\n  {}\n ]\n}}",
        L::B, L::LAT, L::C, L::CIN0, L::COUT, L::MARGIN_PX, L::IN_BYTES, L::OUT_BYTES, L::W_BYTES, L::ACT_BYTES, L::ZERO_BLOCKS, L::L0,
        L::IN_BYTES, L::OUT_BYTES, L::W_BYTES, L::ACT_BYTES,
        res.join(", "),
        maps.join(", "),
        convs.join(",\n  ")
    )
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "taesd".into());
    match which.as_str() {
        "taesd" => println!("{}", taesd_json()),
        other => {
            eprintln!("不认得的程序：{other}（现有：taesd）");
            std::process::exit(2);
        }
    }
}
