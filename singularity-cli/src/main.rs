use clap::{Parser, Subcommand};
use singularity_core::{SystemConfig, TargetArch};
use singularity_cuda::{detect_hardware, validate_precision_and_fallback};
use singularity_server::{create_router, AppState};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

#[derive(Parser)]
#[command(name = "singularity")]
#[command(about = "Project SINGULARITY: High-Performance Rust GPU Inference Substrate", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the OpenAI-compatible REST & SSE inference server
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        #[arg(long, default_value_t = 8080)]
        port: u16,

        #[arg(long, default_value = "./storage")]
        storage_dir: PathBuf,

        #[arg(long)]
        model_path: Option<PathBuf>,

        #[arg(long)]
        tokenizer_path: Option<PathBuf>,

        #[arg(long, default_value_t = 0)]
        device_id: usize,

        #[arg(long)]
        arch: Option<String>,

        #[arg(long)]
        use_cuda: bool,

        /// Explicitly enable the experimental host-only server
        #[arg(long)]
        dev_mock: bool,
    },
    /// Inspect target hardware capabilities (SM100 B200 vs SM90 H200)
    InspectHardware {
        #[arg(long, default_value_t = 0)]
        device_id: i32,

        #[arg(long)]
        arch: Option<String>,
    },
    /// Run offline inference engine benchmarks
    Benchmark {
        #[arg(long, default_value_t = 64)]
        batch_size: usize,

        #[arg(long, default_value_t = 20)]
        steps: usize,
    },
    /// Validate configuration against hardware capabilities
    ValidateConfig {
        #[arg(long)]
        config_path: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,singularity=debug".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            host,
            port,
            storage_dir,
            model_path,
            tokenizer_path,
            device_id,
            arch,
            use_cuda,
            dev_mock,
        } => {
            if use_cuda {
                return Err("CUDA model execution is not implemented: the current transformer forward pass runs on the host".into());
            }
            if !dev_mock {
                return Err("Production inference is not implemented; pass --dev-mock to explicitly start the host-only development server".into());
            }
            tracing::warn!("Development-only server: host inference and mock accelerator; not V4.1 or production GPU execution");
            let mut config = SystemConfig::default();
            config.server.host = host.clone();
            config.server.port = port;

            info!("Initializing Project SINGULARITY server on {host}:{port}...");

            #[cfg(feature = "cuda")]
            let gpu_backend: std::sync::Arc<dyn singularity_cuda::GpuBackend> = if use_cuda {
                let target_arch = match arch.as_deref() {
                    Some("sm90") | Some("h200") => {
                        singularity_cuda::DeviceArchitecture::HopperSm90 {
                            hbm_bytes: 141 * 1024 * 1024 * 1024,
                            nvlink_gen: 4,
                            has_native_fp4: false,
                            tma_version: 1,
                        }
                    }
                    _ => singularity_cuda::DeviceArchitecture::BlackwellSm100 {
                        hbm_bytes: 180 * 1024 * 1024 * 1024,
                        nvlink_gen: 5,
                        has_native_fp4: true,
                        tma_version: 2,
                    },
                };
                info!("Initializing physical CudaBackend on device {device_id} ({target_arch:?})");
                std::sync::Arc::new(singularity_cuda::CudaBackend::new(device_id, target_arch)?)
            } else {
                match arch.as_deref() {
                    Some("h200") | Some("sm90") => {
                        std::sync::Arc::new(singularity_cuda::MockGpuBackend::new_h200())
                    }
                    _ => std::sync::Arc::new(singularity_cuda::MockGpuBackend::new_b200()),
                }
            };

            #[cfg(not(feature = "cuda"))]
            let gpu_backend: std::sync::Arc<dyn singularity_cuda::GpuBackend> = {
                let _ = device_id;
                if use_cuda {
                    tracing::warn!("--use-cuda flag set but binary was compiled without `--features cuda`. Falling back to MockGpuBackend.");
                }
                match arch.as_deref() {
                    Some("h200") | Some("sm90") => {
                        std::sync::Arc::new(singularity_cuda::MockGpuBackend::new_h200())
                    }
                    _ => std::sync::Arc::new(singularity_cuda::MockGpuBackend::new_b200()),
                }
            };

            let tokenizer = if let Some(t_path) = tokenizer_path {
                info!("Loading tokenizer from {}", t_path.display());
                std::sync::Arc::new(singularity_model::TokenizerWrapper::from_file(&t_path)?)
            } else {
                std::sync::Arc::new(singularity_model::TokenizerWrapper::new_fallback())
            };

            let model = if let Some(m_path) = model_path {
                info!("Loading Safetensors model from {}", m_path.display());
                let loader = singularity_model::SafetensorsMmapLoader::open(&m_path)?;
                let model_cfg =
                    if arch.as_deref() == Some("deepseek") || arch.as_deref() == Some("dsv2") {
                        singularity_model::TransformerConfig::deepseek_v2_lite()
                    } else {
                        singularity_model::TransformerConfig::smollm2_135m()
                    };
                std::sync::Arc::new(singularity_model::TransformerModel::from_loader(
                    model_cfg, &loader,
                )?)
            } else {
                let model_cfg =
                    if arch.as_deref() == Some("deepseek") || arch.as_deref() == Some("dsv2") {
                        singularity_model::TransformerConfig::tiny_deepseek()
                    } else {
                        singularity_model::TransformerConfig::tiny()
                    };
                std::sync::Arc::new(singularity_model::TransformerModel::new_random(model_cfg))
            };

            let block_manager =
                std::sync::Arc::new(singularity_cache::PagedBlockManager::with_device_backing(
                    config.cache.total_blocks,
                    config.cache.block_size,
                    gpu_backend.clone(),
                    64,
                )?);

            let state = AppState::new(
                config,
                storage_dir,
                gpu_backend,
                model,
                tokenizer,
                block_manager,
            )?;
            let app = create_router(state);

            let addr: SocketAddr = format!("{host}:{port}").parse()?;
            info!("SINGULARITY listening on http://{addr}");

            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
        }

        Commands::InspectHardware { device_id, arch } => {
            let target_override = match arch.as_deref() {
                Some("sm100") | Some("b200") => Some(TargetArch::BlackwellSm100),
                Some("sm90") | Some("h200") => Some(TargetArch::HopperSm90),
                Some("mock") => Some(TargetArch::HostMock),
                _ => None,
            };

            let detected = detect_hardware(device_id, target_override)?;
            println!("=======================================================");
            println!("   PROJECT SINGULARITY - HARDWARE CAPABILITY REPORT    ");
            println!("=======================================================");
            println!("Device Name:          {}", detected.name());
            println!(
                "HBM3e Capacity:       {:.2} GB",
                detected.hbm_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!("Native NVFP4 Support: {}", detected.has_native_fp4());
            println!(
                "NVLink Generation:    Gen {} ({})",
                detected.nvlink_gen(),
                if detected.nvlink_gen() >= 5 {
                    "1,800 GB/s"
                } else {
                    "900 GB/s"
                }
            );
            println!("TMA Accelerator:      Version {}", detected.tma_version());

            let format = validate_precision_and_fallback(&detected, true)?;
            println!("Selected KV Format:   {format:?}");
            println!("=======================================================");
        }

        Commands::Benchmark { batch_size, steps } => {
            println!(
                "Running host-only mock SINGULARITY microbenchmark (not GPU or V4.1): batch_size={batch_size}, steps={steps}..."
            );
            let mut config = SystemConfig::default();
            config.scheduler.max_batch_size = batch_size;

            let block_manager =
                std::sync::Arc::new(singularity_cache::PagedBlockManager::new(2048, 16));
            let radix_tree = std::sync::Arc::new(singularity_cache::RadixTree::new(10000));
            let gpu_backend: std::sync::Arc<dyn singularity_cuda::GpuBackend> =
                std::sync::Arc::new(singularity_cuda::MockGpuBackend::new_b200());

            let scheduler = singularity_scheduler::ZosScheduler::new(
                config.scheduler.clone(),
                block_manager.clone(),
                radix_tree.clone(),
                gpu_backend.clone(),
            )?;

            // Submit requests
            for i in 0..batch_size {
                let seq = singularity_core::Sequence::new(
                    i as u64 + 1,
                    format!("Benchmark prompt {i}"),
                    vec![101, 102, 103, 104, 105, 106, 107, 108],
                    singularity_core::SamplingParams::default(),
                );
                scheduler.submit_sequence(seq)?;
            }

            let start = Instant::now();
            let mut total_tokens = 0;

            for _ in 0..steps {
                let out = scheduler.step()?;
                for toks in out.generated_tokens.values() {
                    total_tokens += toks.len();
                }
            }

            let elapsed = start.elapsed();
            let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
            let tokens_per_sec = (total_tokens as f64) / elapsed.as_secs_f64();
            let avg_step_ms = elapsed_ms / (steps as f64);

            println!("Benchmark Results:");
            println!("  Total Steps:      {steps}");
            println!("  Total Tokens:     {total_tokens}");
            println!("  Elapsed Time:     {elapsed_ms:.2} ms");
            println!("  Avg Step Latency: {avg_step_ms:.3} ms/step");
            println!("  Throughput:       {tokens_per_sec:.1} tokens/sec");
        }

        Commands::ValidateConfig { config_path } => {
            println!("Validating configuration...");
            let config = if let Some(path) = config_path {
                let content = std::fs::read_to_string(path)?;
                serde_json::from_str::<SystemConfig>(&content)?
            } else {
                SystemConfig::default()
            };

            let detected =
                detect_hardware(config.hardware.device_id, Some(config.hardware.target_arch))?;
            let precision =
                validate_precision_and_fallback(&detected, config.hardware.enable_native_fp4)?;

            println!("Configuration is VALID.");
            println!("  Target Architecture: {:?}", config.hardware.target_arch);
            println!("  Effective Precision: {precision:?}");
            println!("  Max Batch Size:      {}", config.scheduler.max_batch_size);
            println!("  Total KV Blocks:     {}", config.cache.total_blocks);
        }
    }

    Ok(())
}
