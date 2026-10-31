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
    /// Run experimental, single-sequence V4.1 CPU reference inference offline
    V41Reference {
        /// Path to the complete checkpoint directory or safetensors index
        #[arg(long)]
        model_path: PathBuf,
        /// Path to the pinned checkpoint config.json
        #[arg(long)]
        config_path: PathBuf,
        /// Path to the pinned checkpoint tokenizer.json
        #[arg(long)]
        tokenizer_path: PathBuf,
        /// JSON file containing messages, optional tools, mode, effort and max_new_tokens
        #[arg(long)]
        request_path: PathBuf,
        /// Enable sequentially verified DSpark drafts (correctness reference; no speedup)
        #[arg(long)]
        dspark: bool,
    },
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

        /// Path to V4.1 config.json to enable V4.1-Flash-1017 serving
        #[arg(long)]
        v41_config: Option<PathBuf>,

        /// Enable DSpark verified speculative decoding when serving V4.1
        #[arg(long)]
        dspark: bool,

        /// Optional Bearer API key required on /v1/* endpoints
        #[arg(long)]
        api_key: Option<String>,
    },
    /// Verify V4.1 checkpoint inventory and 8x B200 cluster placement plan
    VerifyCheckpoint {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        config_path: PathBuf,
    },
    /// Output the 8x NVIDIA B200 distributed placement and HBM3e budget plan for V4.1
    PlanCluster {
        #[arg(long, default_value_t = 8)]
        world_size: usize,
        #[arg(long, default_value_t = 32)]
        max_sequences: usize,
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

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceRequest {
    messages: Vec<singularity_model::v41_openai::MessageInput>,
    #[serde(default)]
    tools: Vec<singularity_model::v41_openai::ToolDefinition>,
    #[serde(default = "reference_mode")]
    mode: singularity_model::v41_prompt::ThinkingMode,
    #[serde(default = "reference_effort")]
    effort: u8,
    #[serde(default = "reference_max_tokens")]
    max_new_tokens: usize,
}
fn reference_mode() -> singularity_model::v41_prompt::ThinkingMode {
    singularity_model::v41_prompt::ThinkingMode::Chat
}
fn reference_effort() -> u8 {
    75
}
fn reference_max_tokens() -> usize {
    32
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,singularity=debug".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::V41Reference {
            model_path,
            config_path,
            tokenizer_path,
            request_path,
            dspark,
        } => {
            let request: ReferenceRequest =
                serde_json::from_reader(std::fs::File::open(request_path)?)?;
            if request.max_new_tokens == 0 {
                return Err("max_new_tokens must be positive".into());
            }
            let config = singularity_model::v41_config::V41Config::load(config_path)?;
            let tokenizer = singularity_model::TokenizerWrapper::from_v41_file(tokenizer_path)?;
            let prompt = singularity_model::v41_openai::encode(
                &request.messages,
                &request.tools,
                request.mode,
                request.effort,
            )?;
            let tokens = tokenizer.encode(&prompt)?;
            let max_length = tokens
                .len()
                .checked_add(request.max_new_tokens)
                .ok_or("Sequence length overflow")?;
            let required_length = if dspark {
                max_length
                    .checked_add(config.text_config.dspark_block_size)
                    .ok_or("Sequence length overflow")?
            } else {
                max_length
            };
            if tokens.is_empty() || required_length > config.text_config.max_position_embeddings {
                return Err("Requested sequence exceeds checkpoint context length".into());
            }
            eprintln!("Experimental CPU reference: full checkpoint required; execution may be extremely slow and is not production inference.");
            let token_map =
                tokenizer.v41_engram_map(config.text_config.engram_compressed_vocab_size)?;
            let loader = singularity_model::SafetensorsMmapLoader::open(model_path)?;
            let mut model = singularity_model::v41_model::TextModel::load(
                &loader,
                &config.text_config,
                Some(token_map.into()),
                2,
            )?;
            let (generated, dspark_stats) = if dspark {
                let io = singularity_model::v41_io::TokenIo::load(&loader, &config.text_config)?;
                let mut draft =
                    singularity_model::v41_dspark::DraftModel::load(&loader, &config.text_config)?;
                let result = singularity_model::v41_dspark::generate_verified_greedy(
                    &mut model,
                    &mut draft,
                    &io,
                    &tokens,
                    request.max_new_tokens,
                    tokenizer.eos_token(),
                )?;
                (
                    result.tokens,
                    Some(
                        serde_json::json!({"proposed":result.proposed,"accepted":result.accepted}),
                    ),
                )
            } else {
                (
                    model.generate_greedy(
                        &tokens,
                        request.max_new_tokens,
                        tokenizer.eos_token(),
                    )?,
                    None,
                )
            };
            let output = tokenizer.decode(&generated)?;
            let ended = generated.last() == Some(&tokenizer.eos_token());
            let message = if ended {
                let parsed = singularity_model::v41_tools::parse_completion(&output, request.mode)?;
                let ids = (0..parsed.calls.len())
                    .map(|i| format!("call_ref_{i}"))
                    .collect::<Vec<_>>();
                Some(singularity_model::v41_openai::decode(
                    &output,
                    request.mode,
                    &ids,
                )?)
            } else {
                None
            };
            let finish_reason = if !ended {
                "length"
            } else if message.as_ref().is_some_and(|m| !m.tool_calls.is_empty()) {
                "tool_calls"
            } else {
                "stop"
            };
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"prompt_tokens":tokens.len(),"completion_tokens":generated,"finish_reason":finish_reason,"text":output,"message":message,"dspark":dspark_stats})
                )?
            );
        }
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
            v41_config,
            dspark,
            api_key,
        } => {
            if v41_config.is_none() && !dev_mock {
                return Err("Pass --v41-config <path> --model-path <path> to serve DeepSeek-V4.1-Flash-1017, or --dev-mock for the host development server".into());
            }
            if v41_config.is_none() && use_cuda {
                return Err("Host development model does not run on CUDA; pass --v41-config <path> for V4.1 serving".into());
            }
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

            let tokenizer = if let Some(ref t_path) = tokenizer_path {
                info!("Loading tokenizer from {}", t_path.display());
                std::sync::Arc::new(
                    singularity_model::TokenizerWrapper::from_v41_file(t_path)
                        .or_else(|_| singularity_model::TokenizerWrapper::from_file(t_path))?,
                )
            } else {
                std::sync::Arc::new(singularity_model::TokenizerWrapper::new_fallback())
            };

            let model = if v41_config.is_none() {
                if let Some(ref m_path) = model_path {
                    info!("Loading Safetensors model from {}", m_path.display());
                    let loader = singularity_model::SafetensorsMmapLoader::open(m_path)?;
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
                }
            } else {
                std::sync::Arc::new(singularity_model::TransformerModel::new_random(
                    singularity_model::TransformerConfig::tiny(),
                ))
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

            if let Some(key) = api_key {
                state.set_api_key(Some(key));
            }

            if v41_config.is_some() {
                state.set_readiness(singularity_server::ReadinessState {
                    container_alive: true,
                    runtime_initialized: true,
                    cuda_operational: true,
                    gpus_available: true,
                    distributed_topology_operational: true,
                    checkpoint_present: true,
                    model_loaded: false,
                    model_ready: false,
                    inference_executed: false,
                });
            }

            let app = create_router(state.clone());
            let addr: SocketAddr = format!("{host}:{port}").parse()?;
            let listener = tokio::net::TcpListener::bind(addr).await?;
            info!("SINGULARITY bound HTTP listener on http://{addr}");
            let serve_handle = tokio::spawn(async move {
                axum::serve(listener, app)
                    .with_graceful_shutdown(shutdown_signal())
                    .await
            });

            if let Some(cfg_path) = v41_config {
                let m_path =
                    model_path.ok_or("--model-path is required when --v41-config is specified")?;
                info!(
                    "Loading V4.1-Flash-1017 engine from {} with config {} (dspark={})...",
                    m_path.display(),
                    cfg_path.display(),
                    dspark
                );
                let t_path = tokenizer_path.clone();
                let engine = tokio::task::spawn_blocking(move || {
                    singularity_server::V41ServingEngine::load_from_paths(
                        &m_path,
                        &cfg_path,
                        t_path.as_deref(),
                        dspark,
                    )
                })
                .await??;
                state.set_v41_engine(engine);
                state.set_readiness(singularity_server::ReadinessState::default_ready());
                info!("V4.1-Flash-1017 warm-up inference succeeded; server marked READY.");
            } else {
                tracing::warn!("Development-only server: host inference and mock accelerator; not V4.1 or production GPU execution");
            }

            serve_handle.await??;
        }

        Commands::VerifyCheckpoint {
            model_path,
            config_path,
        } => {
            let config = singularity_model::v41_config::V41Config::load(&config_path)?;
            let loader = singularity_model::SafetensorsMmapLoader::open(&model_path)?;
            let summary = config.validate_checkpoint_inventory(&loader)?;
            let cluster_plan = singularity_cuda::ClusterPlacementPlan::plan_v41_cluster(
                singularity_cuda::B200ClusterSpec::default(),
                singularity_cuda::V41ClusterTopology::default(),
                32,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "valid",
                    "verified_tensors": summary.verified_tensors,
                    "backbone_layers": summary.backbone_layers,
                    "draft_stages": summary.draft_stages,
                    "vision_layers": summary.vision_layers,
                    "cluster_world_size": cluster_plan.spec.world_size,
                    "rank0_allocated_bytes": cluster_plan.ranks[0].total_allocated_bytes,
                    "rank0_headroom_bytes": cluster_plan.ranks[0].headroom_bytes,
                }))?
            );
        }

        Commands::PlanCluster {
            world_size,
            max_sequences,
        } => {
            let spec = singularity_cuda::B200ClusterSpec {
                world_size,
                ..singularity_cuda::B200ClusterSpec::default()
            };
            let plan = singularity_cuda::ClusterPlacementPlan::plan_v41_cluster(
                spec,
                singularity_cuda::V41ClusterTopology::default(),
                max_sequences,
            )?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
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

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sigterm) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sigterm.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received SIGINT; initiating graceful HTTP server shutdown...");
        }
        _ = terminate => {
            info!("Received SIGTERM; initiating graceful HTTP server shutdown...");
        }
    }
}
