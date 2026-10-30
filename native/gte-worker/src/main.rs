//! Persistent CPU INT8 GTE-tiny worker. stdout is exclusively JSON-lines.
use anyhow::{anyhow, bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use secp256k1::{schnorr::Signature, Secp256k1, XOnlyPublicKey};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

const MODEL: &str =
    "TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3;onnx-int8-ceab9964bcdb";
const MODEL_SHA: &str = "ceab9964bcdb19ad013a2fe44bc1ddb3657ab0af1f48cddc349014d6085762ca";
const TOKENIZER_SHA: &str = "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0";
const DIM: usize = 384;
const FRAME_LIMIT: usize = 8 * 1024 * 1024;

struct Options {
    bundle: PathBuf,
    library: Option<PathBuf>,
    threads: usize,
    batch: usize,
}
impl Options {
    fn parse() -> Result<Self> {
        let mut out = Self {
            bundle: std::env::var_os("MONDRIAN_GTE_BUNDLE")
                .map(PathBuf::from)
                .unwrap_or_else(|| "output/gte_tiny_rust".into()),
            library: None,
            threads: 2,
            batch: 8,
        };
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            if key == "--help" {
                eprintln!("mondrian-gte-worker [--bundle DIR] [--ort-library FILE] [--threads N] [--batch-size 1..32] [--device cpu] [--model ID]");
                std::process::exit(0);
            }
            let value = args.next().context("missing option value")?;
            match key.as_str() {
                "--bundle" => out.bundle = value.into(),
                "--ort-library" => out.library = Some(value.into()),
                "--threads" => out.threads = value.parse()?,
                "--batch-size" => out.batch = value.parse()?,
                "--device" => ensure!(value == "cpu", "this worker supports CPU only"),
                "--model" => ensure!(value == MODEL, "unsupported model identity"),
                _ => bail!("unknown option: {key}"),
            }
        }
        ensure!(
            (1..=32).contains(&out.batch) && (1..=256).contains(&out.threads),
            "invalid batch/thread count"
        );
        Ok(out)
    }
}

fn check_hash(path: &Path, expected: &str) -> Result<()> {
    let mut file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    ensure!(
        format!("{:x}", hash.finalize()) == expected,
        "checksum mismatch: {}",
        path.display()
    );
    Ok(())
}

struct Encoder {
    tokenizer: Tokenizer,
    session: Session,
    batch: usize,
}
impl Encoder {
    fn load(opts: &Options) -> Result<Self> {
        let model = opts.bundle.join("model_quantized.onnx");
        let tokenizer_path = opts.bundle.join("tokenizer.json");
        check_hash(&model, MODEL_SHA)?;
        check_hash(&tokenizer_path, TOKENIZER_SHA)?;
        let library = opts
            .library
            .clone()
            .unwrap_or_else(|| opts.bundle.join("libonnxruntime.so"));
        let env = ort::init_from(library)?
            .with_name("mondrian-gte")
            .with_telemetry(false)
            .build()?;
        let session = Session::builder(&env)?
            .with_intra_threads(opts.threads)
            .map_err(|e| anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(model)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("{e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 256,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("{e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_id: 0,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }));
        Ok(Self {
            tokenizer,
            session,
            batch: opts.batch,
        })
    }

    fn tokens(&self, texts: &[String]) -> Result<Vec<tokenizers::Encoding>> {
        // No Rayon pool: ORT owns the configured inference thread budget.
        self.tokenizer
            .encode_batch(texts.iter().map(String::as_str).collect(), true)
            .map_err(|_| anyhow!("tokenization failed"))
    }

    fn encode(&mut self, texts: &[String]) -> Result<Vec<f32>> {
        let mut vectors = Vec::with_capacity(texts.len() * DIM);
        for batch in texts.chunks(self.batch) {
            let tokens = self.tokens(batch)?;
            let seq = tokens[0].len();
            let shape = [batch.len(), seq];
            let ids: Vec<i64> = tokens
                .iter()
                .flat_map(|t| t.get_ids().iter().map(|&x| i64::from(x)))
                .collect();
            let mask: Vec<i64> = tokens
                .iter()
                .flat_map(|t| t.get_attention_mask().iter().map(|&x| i64::from(x)))
                .collect();
            let types: Vec<i64> = tokens
                .iter()
                .flat_map(|t| t.get_type_ids().iter().map(|&x| i64::from(x)))
                .collect();
            let result = self.session.run(ort::inputs![
                "input_ids" => Tensor::from_array((shape, ids))?,
                "attention_mask" => Tensor::from_array((shape, mask.clone()))?,
                "token_type_ids" => Tensor::from_array((shape, types))?
            ])?;
            let (dims, hidden) = result["last_hidden_state"].try_extract_tensor::<f32>()?;
            ensure!(
                dims[..] == [batch.len() as i64, seq as i64, DIM as i64],
                "invalid hidden-state shape"
            );
            for b in 0..batch.len() {
                let mut pooled = [0f32; DIM];
                let mut count = 0f32;
                for t in 0..seq {
                    if mask[b * seq + t] != 0 {
                        count += 1.;
                        let offset = (b * seq + t) * DIM;
                        for d in 0..DIM {
                            pooled[d] += hidden[offset + d];
                        }
                    }
                }
                ensure!(count > 0., "empty attention mask");
                for x in &mut pooled {
                    *x /= count;
                }
                let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
                ensure!(norm.is_finite() && norm > 0., "invalid output norm");
                for x in &mut pooled {
                    *x /= norm;
                }
                ensure!(pooled.iter().all(|x| x.is_finite()), "nonfinite embedding");
                vectors.extend_from_slice(&pooled);
            }
        }
        Ok(vectors)
    }
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Embed { texts: Vec<String> },
    Tokenize { texts: Vec<String> },
    VerifyNostr { events: Vec<Value> },
    Close,
}

fn check_texts(texts: &[String]) -> Result<()> {
    ensure!(
        !texts.is_empty() && texts.len() <= 1024 && texts.iter().all(|t| t.len() <= 65536),
        "expected 1..1024 texts, each at most 64 KiB"
    );
    Ok(())
}

fn hex_bytes<const N: usize>(value: &Value) -> Option<[u8; N]> {
    let s = value.as_str()?;
    if s.len() != N * 2
        || !s
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return None;
    }
    let mut out = [0; N];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

fn verify_event(event: &Value) -> bool {
    let checked = || -> Option<bool> {
        let id = hex_bytes::<32>(&event["id"])?;
        let key = XOnlyPublicKey::from_byte_array(hex_bytes::<32>(&event["pubkey"])?).ok()?;
        let sig = Signature::from_byte_array(hex_bytes::<64>(&event["sig"])?);
        let kind = event["kind"].as_u64()?;
        if kind > 65535 {
            return None;
        }
        let at = event["created_at"].as_u64()?;
        let content = event["content"].as_str()?;
        let tags = event["tags"].as_array()?;
        if !tags
            .iter()
            .all(|t| t.as_array().is_some_and(|a| a.iter().all(Value::is_string)))
        {
            return None;
        }
        let payload =
            serde_json::to_vec(&json!([0, event["pubkey"], at, kind, tags, content])).ok()?;
        let digest: [u8; 32] = Sha256::digest(payload).into();
        Some(
            digest == id
                && Secp256k1::verification_only()
                    .verify_schnorr(&sig, &digest, &key)
                    .is_ok(),
        )
    };
    checked().unwrap_or(false)
}

fn respond(enc: &mut Encoder, req: Request) -> Result<Value> {
    match req {
        Request::Embed { texts } => {
            check_texts(&texts)?;
            let vectors = enc.encode(&texts)?;
            let bytes: Vec<u8> = vectors.iter().flat_map(|x| x.to_le_bytes()).collect();
            Ok(
                json!({"model": MODEL, "dimension": DIM, "count": texts.len(), "provider": "CPUExecutionProvider",
                "vectors": STANDARD.encode(bytes)}),
            )
        }
        Request::Tokenize { texts } => {
            check_texts(&texts)?;
            let tokens = enc.tokens(&texts)?;
            Ok(
                json!({"ids": tokens.iter().map(|t| t.get_ids()).collect::<Vec<_>>(),
                "mask": tokens.iter().map(|t| t.get_attention_mask()).collect::<Vec<_>>()}),
            )
        }
        Request::VerifyNostr { events } => {
            ensure!(events.len() <= 1024, "too many events");
            Ok(json!({"valid": events.iter().map(verify_event).collect::<Vec<_>>()}))
        }
        Request::Close => Ok(json!({"ok": true})),
    }
}

// A bounded read, including malformed frames: no unbounded read_line allocation.
fn read_frame(input: &mut impl BufRead, buf: &mut Vec<u8>) -> io::Result<bool> {
    buf.clear();
    let n = input
        .take((FRAME_LIMIT + 1) as u64)
        .read_until(b'\n', buf)?;
    if buf.len() > FRAME_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request exceeds 8 MiB",
        ));
    }
    Ok(n != 0)
}
fn send(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn run() -> Result<()> {
    let opts = Options::parse()?;
    tokenizers::utils::parallelism::set_parallelism(false);
    let mut enc = Encoder::load(&opts)?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    send(
        &mut out,
        &json!({"ready": true, "model": MODEL, "dimension": DIM,
        "provider": "CPUExecutionProvider", "precision": "int8", "threads": opts.threads,
        "batch_size": opts.batch, "runtime": ort::info(), "model_sha256": MODEL_SHA}),
    )?;
    let mut input = io::stdin().lock();
    let mut buf = Vec::new();
    while read_frame(&mut input, &mut buf)? {
        let req = serde_json::from_slice::<Request>(&buf);
        let closing = matches!(req, Ok(Request::Close));
        let response = match req {
            Ok(req) => respond(&mut enc, req)
                .unwrap_or_else(|_| json!({"error": "invalid request or inference failure"})),
            Err(_) => json!({"error": "invalid request"}),
        };
        send(&mut out, &response)?;
        if closing {
            break;
        }
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("gte worker: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_frames() {
        let mut buf = Vec::new();
        let data = vec![b'x'; FRAME_LIMIT + 2];
        assert!(read_frame(&mut &data[..], &mut buf).is_err());
        assert_eq!(buf.len(), FRAME_LIMIT + 1);
        let mut input = &b"{}\n{}\n"[..];
        assert!(read_frame(&mut input, &mut buf).unwrap());
        assert_eq!(buf, b"{}\n");
        assert!(read_frame(&mut input, &mut buf).unwrap());
        assert!(!read_frame(&mut input, &mut buf).unwrap());
    }
    #[test]
    fn rejects_invalid_events_and_requests() {
        for e in [json!(null), json!({}), json!({"pubkey": "A".repeat(64)})] {
            assert!(!verify_event(&e));
        }
        assert!(check_texts(&[]).is_err());
        assert!(check_texts(&vec!["x".into(); 1025]).is_err());
        assert!(check_texts(&["🦋".repeat(16385)]).is_err());
        assert!(serde_json::from_value::<Request>(json!({"op":"embed","texts":[1]})).is_err());
    }
}
