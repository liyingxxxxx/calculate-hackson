//! `layout-dump <taesd|vae>`：把 `crates/<程序>/src/layout.rs` 打成 JSON（stdout）。
//!
//! 为什么不让 Python 自己算偏移：设备程序（Rust、编译期常量）与主机打包要是各写一份，改一边忘一边就是静默的错数。
//! 这里用 `#[path]` 把设备 crate 的 layout.rs 原样编进主机程序（同 calBLAS `examples/tiny/pack` 的做法），Python 只读这份 JSON。
//! 不引第三方 crate（serde 之类）：JSON 手写，键都是固定的。
#[path = "../../taesd/src/layout.rs"]
#[allow(dead_code)]
mod taesd;
#[path = "../../vae/src/layout.rs"]
#[allow(dead_code)]
mod vae;

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

fn vae_json() -> String {
    use vae as L;
    let convs: Vec<String> = L::CONVS
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "{{\"i\": {i}, \"key\": \"{}\", \"cin\": {}, \"cout\": {}, \"cin_pad\": {}, \"cout_pad\": {}, \"k\": {}, \"res\": {}, \"linear\": {}, \"fold\": {}, \"w_len\": {}, \"b_len\": {}}}",
                c.key, c.cin, c.cout, c.cin_pad, c.cout_pad, c.k, c.res, c.linear, c.fold, L::w_len(i), L::b_len(i)
            )
        })
        .collect();
    let gns: Vec<String> = L::GNS
        .iter()
        .enumerate()
        .map(|(j, g)| format!("{{\"j\": {j}, \"key\": \"{}\", \"c\": {}, \"res\": {}, \"silu\": {}, \"nhwc\": {}, \"cols\": {}, \"len\": {}}}", g.key, g.c, g.res, g.silu, g.nhwc, L::gn_cols(j), L::gn_len(j)))
        .collect();
    let maps: Vec<String> = L::MAPS.iter().map(|m| format!("{{\"res\": {}, \"ch\": {}}}", m.res, m.ch)).collect();
    let mut stages = Vec::new();
    for (s, st) in L::STAGES.iter().enumerate() {
        let mo: Vec<String> = st.maps.iter().map(|&i| format!("\"{i}\": {}", L::map_off(s, i))).collect();
        let cw: Vec<String> = (st.c0..st.c1).map(|i| format!("\"{i}\": [{}, {}]", L::w_off(s, i), L::b_off(s, i))).collect();
        let gw: Vec<String> = (st.g0..st.g1).map(|j| format!("\"{j}\": [{}, {}]", L::gw_off(s, j), L::gb_off(s, j))).collect();
        let masks: Vec<String> = (0..3).filter(|&m| L::uses_mask(s, m)).map(|m| format!("\"{}\": {}", L::MASK_C[m], L::mask_off(s, L::MASK_C[m]))).collect();
        let (tok, qkv) = if st.attn {
            (
                format!("[{}]", (0..5).map(|t| L::tok(s, t).to_string()).collect::<Vec<_>>().join(", ")),
                format!("[{}]", (0..3).map(|q| format!("[{}, {}]", L::qkv_w_off(s, q), L::qkv_b_off(s, q))).collect::<Vec<_>>().join(", ")),
            )
        } else {
            ("null".to_string(), "null".to_string())
        };
        stages.push(format!(
            "{{\"s\": {s}, \"name\": \"{}\", \"c0\": {}, \"c1\": {}, \"g0\": {}, \"g1\": {}, \"attn\": {}, \"inp\": [{}, {}], \"out\": [{}, {}], \"in_bytes\": {}, \"out_bytes\": {}, \"w_bytes\": {}, \"act_bytes\": {}, \"maps_bytes\": {}, \"zero_blocks\": {}, \"st\": {}, \"n1\": {}, \"n2\": {}, \"nchw_bytes\": {}, \"tok\": {tok}, \"maps\": {{{}}}, \"convs\": {{{}}}, \"gns\": {{{}}}, \"masks\": {{{}}}, \"qkv\": {qkv}}}",
            st.name, st.c0, st.c1, st.g0, st.g1, st.attn, st.inp.res, st.inp.ch, st.out.res, st.out.ch,
            L::in_bytes(s), L::out_bytes(s), L::w_bytes(s), L::act_bytes(s), L::maps_bytes(s), L::zero_blocks(s), L::st(s), L::n1(s), L::n2(s), L::nchw_bytes(s),
            mo.join(", "), cw.join(", "), gw.join(", "), masks.join(", ")
        ));
    }
    format!(
        "{{\n \"program\": \"vae\",\n \"consts\": {{\"B\": {}, \"LAT\": {}, \"CIN0\": {}, \"COUT\": {}, \"MARGIN_PX\": {}, \"GROUPS\": {}, \"NT\": {}, \"DA\": {}, \"IN_BYTES\": {}, \"OUT_BYTES\": {}, \"QKV_W_LEN\": {}, \"QKV_B_LEN\": {}, \"TOK_BYTES\": {}, \"NSTAGE\": {}}},\n \"maps\": [{}],\n \"convs\": [\n  {}\n ],\n \"gns\": [\n  {}\n ],\n \"qkv_keys\": [{}],\n \"stages\": [\n  {}\n ]\n}}",
        L::B, L::LAT, L::CIN0, L::COUT, L::MARGIN_PX, L::GROUPS, L::NT, L::DA, L::IN_BYTES, L::OUT_BYTES, L::QKV_W_LEN, L::QKV_B_LEN, L::TOK_BYTES, L::NSTAGE,
        maps.join(", "),
        convs.join(",\n  "),
        gns.join(",\n  "),
        L::QKV_KEYS.iter().map(|k| format!("\"{k}\"")).collect::<Vec<_>>().join(", "),
        stages.join(",\n  ")
    )
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "taesd".into());
    match which.as_str() {
        "taesd" => println!("{}", taesd_json()),
        "vae" => println!("{}", vae_json()),
        other => {
            eprintln!("不认得的程序：{other}（现有：taesd、vae）");
            std::process::exit(2);
        }
    }
}
