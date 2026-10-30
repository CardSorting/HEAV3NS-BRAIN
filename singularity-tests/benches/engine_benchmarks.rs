use criterion::{black_box, criterion_group, criterion_main, Criterion};
use singularity_tests::*;
use std::sync::Arc;

fn bench_block_allocation(c: &mut Criterion) {
    c.bench_function("paged_block_allocation_and_free_100", |b| {
        let manager = PagedBlockManager::new(1024, 16);
        b.iter(|| {
            let blocks = manager.allocate_blocks(black_box(16)).unwrap();
            manager.free_blocks(&blocks).unwrap();
        });
    });
}

fn bench_radix_prefix_cache(c: &mut Criterion) {
    let block_mgr = PagedBlockManager::new(1024, 16);
    let radix = RadixTree::new(10000);
    let prefix = vec![101, 102, 103, 104, 105, 106, 107, 108];
    let blocks = block_mgr.allocate_blocks(2).unwrap();
    radix.insert_prefix(&prefix, &blocks, &block_mgr).unwrap();

    c.bench_function("radix_tree_prefix_match_hit", |b| {
        b.iter(|| {
            let res = radix.match_prefix(black_box(&prefix));
            black_box(res);
        });
    });
}

fn bench_lcfs_sorting_and_preemption(c: &mut Criterion) {
    let block_mgr = Arc::new(PagedBlockManager::new(4, 16));
    c.bench_function("lcfs_schedule_and_preemption_64_seqs", |b| {
        b.iter(|| {
            let mut running = Vec::with_capacity(64);
            for i in 0..64 {
                let mut seq = Sequence::new(
                    i as u64,
                    format!("p_{i}"),
                    vec![1; 16],
                    SamplingParams::default(),
                );
                for j in 0..i {
                    seq.output_tokens.push(j as u32);
                }
                running.push(seq);
            }
            let res = schedule_decode_lcfs(running, black_box(32), &block_mgr).unwrap();
            black_box(res);
        });
    });
}

fn bench_kmp_and_grammar_bitmask(c: &mut Criterion) {
    let engine = GrammarBitmaskEngine::new(32000);
    let mut logits = vec![1.0f32; 32000];

    c.bench_function("grammar_bitmask_apply_32k_vocab", |b| {
        b.iter(|| {
            engine
                .apply_mask(&mut logits, black_box(JsonGrammarState::ExpectObjectStart))
                .unwrap();
            black_box(&logits);
        });
    });
}

fn bench_zos_scheduler_step(c: &mut Criterion) {
    let config = SchedulerConfig {
        max_batch_size: 64,
        ..Default::default()
    };
    let block_mgr = Arc::new(PagedBlockManager::new(1024, 16));
    let radix = Arc::new(RadixTree::new(5000));
    let gpu = Arc::new(MockGpuBackend::new_b200());
    let scheduler = ZosScheduler::new(config, block_mgr, radix, gpu).unwrap();

    for i in 0..32 {
        let seq = Sequence::new(
            i as u64,
            format!("p_{i}"),
            vec![1, 2, 3, 4],
            SamplingParams::default(),
        );
        scheduler.submit_sequence(seq).unwrap();
    }

    c.bench_function("zos_scheduler_step_32_seqs", |b| {
        b.iter(|| {
            let out = scheduler.step().unwrap();
            black_box(out);
        });
    });
}

criterion_group!(
    benches,
    bench_block_allocation,
    bench_radix_prefix_cache,
    bench_lcfs_sorting_and_preemption,
    bench_kmp_and_grammar_bitmask,
    bench_zos_scheduler_step
);
criterion_main!(benches);
