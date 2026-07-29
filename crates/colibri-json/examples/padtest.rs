fn main() {
    let with_pad = "{\"a\":1}          ";
    let r = colibri_json::Json::parse(with_pad);
    println!("trailing-space padding parses: {}", r.is_some());
    let with_meta = "{\"__metadata__\":{\"pad\":\"xxxxx\"},\"a\":1}";
    println!("__metadata__ padding parses: {}", colibri_json::Json::parse(with_meta).is_some());
}
