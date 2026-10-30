use crate::block_manager::PagedBlockManager;
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use singularity_core::SingularityError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct RadixNode {
    pub tokens: Vec<u32>,
    pub block_ids: Vec<usize>,
    pub children: HashMap<u32, Arc<RwLock<RadixNode>>>,
    pub ref_count: usize,
    pub generation: u64,
    pub sha256_hash: [u8; 32],
    pub last_accessed: Instant,
}

impl RadixNode {
    pub fn new(tokens: Vec<u32>, block_ids: Vec<usize>, generation: u64) -> Self {
        let mut hasher = Sha256::new();
        for &t in &tokens {
            hasher.update(t.to_le_bytes());
        }
        let sha256_hash = hasher.finalize().into();

        Self {
            tokens,
            block_ids,
            children: HashMap::new(),
            ref_count: 0,
            generation,
            sha256_hash,
            last_accessed: Instant::now(),
        }
    }

    pub fn compute_hash(tokens: &[u32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for &t in tokens {
            hasher.update(t.to_le_bytes());
        }
        hasher.finalize().into()
    }
}

pub struct RadixTree {
    root: Arc<RwLock<RadixNode>>,
    generation_counter: AtomicU64,
    #[allow(dead_code)]
    max_nodes: usize,
}

impl RadixTree {
    pub fn new(max_nodes: usize) -> Self {
        Self {
            root: Arc::new(RwLock::new(RadixNode::new(Vec::new(), Vec::new(), 0))),
            generation_counter: AtomicU64::new(1),
            max_nodes,
        }
    }

    /// Matches the longest common prefix for the given prompt tokens.
    /// Returns: (matched_block_ids, num_tokens_matched)
    pub fn match_prefix(&self, tokens: &[u32]) -> (Vec<usize>, usize) {
        if tokens.is_empty() {
            return (Vec::new(), 0);
        }

        let mut current_node = Arc::clone(&self.root);
        let mut matched_blocks = Vec::new();
        let mut token_offset = 0;

        loop {
            let guard = current_node.read();
            if token_offset >= tokens.len() {
                break;
            }

            let next_token = tokens[token_offset];
            if let Some(child) = guard.children.get(&next_token) {
                let next_node = Arc::clone(child);
                drop(guard);

                let mut child_guard = next_node.write();
                child_guard.last_accessed = Instant::now();

                // Check how many tokens in this child match
                let node_tokens = &child_guard.tokens;
                let mut match_len = 0;
                while match_len < node_tokens.len()
                    && token_offset + match_len < tokens.len()
                    && node_tokens[match_len] == tokens[token_offset + match_len]
                {
                    match_len += 1;
                }

                if match_len == node_tokens.len() {
                    // Full match of this node
                    matched_blocks.extend_from_slice(&child_guard.block_ids);
                    token_offset += match_len;
                    drop(child_guard);
                    current_node = next_node;
                } else {
                    // Partial match inside node tokens: do not add partial block IDs
                    break;
                }
            } else {
                break;
            }
        }

        (matched_blocks, token_offset)
    }

    /// Inserts prefix tokens and their allocated block IDs into the radix trie.
    pub fn insert_prefix(
        &self,
        tokens: &[u32],
        block_ids: &[usize],
        block_manager: &PagedBlockManager,
    ) -> Result<(), SingularityError> {
        if tokens.is_empty() || block_ids.is_empty() {
            return Ok(());
        }

        let gen = self.generation_counter.fetch_add(1, Ordering::SeqCst);
        let mut current_node = Arc::clone(&self.root);
        let mut token_offset = 0;
        let mut block_offset = 0;

        while token_offset < tokens.len() {
            let first_token = tokens[token_offset];
            let remaining_tokens = &tokens[token_offset..];
            let remaining_blocks = &block_ids[block_offset..];

            let mut guard = current_node.write();
            if let Some(child_arc) = guard.children.get(&first_token).cloned() {
                drop(guard);

                let mut child = child_arc.write();
                child.last_accessed = Instant::now();
                child.generation = gen;

                // Find common prefix length between child.tokens and remaining_tokens
                let mut common_len = 0;
                while common_len < child.tokens.len()
                    && common_len < remaining_tokens.len()
                    && child.tokens[common_len] == remaining_tokens[common_len]
                {
                    common_len += 1;
                }

                if common_len == child.tokens.len() {
                    // Full match of this child node, continue down tree
                    token_offset += common_len;
                    block_offset = (block_offset + child.block_ids.len()).min(block_ids.len());
                    drop(child);
                    current_node = child_arc;
                } else {
                    // Partial match: split child node at common_len
                    let child_remaining_tokens = child.tokens[common_len..].to_vec();

                    let split_block_idx = if child.tokens.is_empty() {
                        0
                    } else if block_manager.block_size() > 0
                        && common_len >= block_manager.block_size()
                    {
                        (common_len / block_manager.block_size()).min(child.block_ids.len())
                    } else {
                        ((common_len * child.block_ids.len()) / child.tokens.len())
                            .min(child.block_ids.len())
                    };

                    let split_blocks = child.block_ids[split_block_idx..].to_vec();
                    let parent_blocks = child.block_ids[0..split_block_idx].to_vec();

                    let split_first_token = child_remaining_tokens[0];
                    let mut split_node = RadixNode::new(child_remaining_tokens, split_blocks, gen);
                    split_node.children = std::mem::take(&mut child.children);
                    split_node.ref_count = child.ref_count;

                    child.tokens.truncate(common_len);
                    child.block_ids = parent_blocks;
                    child.sha256_hash = RadixNode::compute_hash(&child.tokens);
                    child
                        .children
                        .insert(split_first_token, Arc::new(RwLock::new(split_node)));

                    let new_tokens_remaining = &remaining_tokens[common_len..];
                    if !new_tokens_remaining.is_empty() {
                        let new_blocks_for_branch = if remaining_blocks.len() > split_block_idx {
                            remaining_blocks[split_block_idx..].to_vec()
                        } else {
                            Vec::new()
                        };
                        for &bid in &new_blocks_for_branch {
                            block_manager.mark_prefix_cached(bid, true);
                        }
                        let new_branch_first = new_tokens_remaining[0];
                        let new_branch_node = RadixNode::new(
                            new_tokens_remaining.to_vec(),
                            new_blocks_for_branch,
                            gen,
                        );
                        child
                            .children
                            .insert(new_branch_first, Arc::new(RwLock::new(new_branch_node)));
                    }

                    return Ok(());
                }
            } else {
                let new_node =
                    RadixNode::new(remaining_tokens.to_vec(), remaining_blocks.to_vec(), gen);
                for &bid in remaining_blocks {
                    block_manager.mark_prefix_cached(bid, true);
                }
                guard
                    .children
                    .insert(first_token, Arc::new(RwLock::new(new_node)));
                return Ok(());
            }
        }

        Ok(())
    }

    /// Evicts unpinned prefix nodes in LRU order to recover blocks when needed.
    pub fn evict_lru(&self, block_manager: &PagedBlockManager, target_free: usize) -> usize {
        let mut freed_blocks = 0;
        let mut root_guard = self.root.write();

        let mut candidates: Vec<(u32, Instant)> = Vec::new();
        for (&k, child) in root_guard.children.iter() {
            let guard = child.read();
            let any_pinned = guard
                .block_ids
                .iter()
                .any(|&bid| block_manager.is_block_pinned(bid));
            if !any_pinned && guard.ref_count == 0 {
                candidates.push((k, guard.last_accessed));
            }
        }

        candidates.sort_by_key(|&(_, time)| time);

        for (key, _) in candidates {
            if block_manager.free_block_count() >= target_free {
                break;
            }
            if let Some(removed_node) = root_guard.children.remove(&key) {
                let guard = removed_node.read();
                freed_blocks += free_node_recursive(&guard, block_manager);
            }
        }

        freed_blocks
    }

    pub fn node_count(&self) -> usize {
        self.root.read().children.len()
    }
}

fn free_node_recursive(node: &RadixNode, block_manager: &PagedBlockManager) -> usize {
    let mut count = 0;
    for &bid in &node.block_ids {
        block_manager.mark_prefix_cached(bid, false);
        if block_manager.is_block_allocated(bid) {
            let _ = block_manager.free_block(bid);
            count += 1;
        }
    }
    for child in node.children.values() {
        let child_guard = child.read();
        count += free_node_recursive(&child_guard, block_manager);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::PagedBlockManager;

    #[test]
    fn test_radix_tree_splitting_and_prefix_match() {
        let block_manager = PagedBlockManager::new(64, 16);
        let radix = RadixTree::new(1000);

        let prefix_a = vec![101, 102, 103, 104];
        let blocks_a = block_manager.allocate_blocks(2).unwrap();
        radix
            .insert_prefix(&prefix_a, &blocks_a, &block_manager)
            .unwrap();

        let (matched_blocks, matched_len) = radix.match_prefix(&prefix_a);
        assert_eq!(matched_len, 4);
        assert_eq!(matched_blocks, blocks_a);

        // Insert branching prefix B sharing [101, 102]
        let prefix_b = vec![101, 102, 200, 201];
        let blocks_b = block_manager.allocate_blocks(2).unwrap();
        radix
            .insert_prefix(&prefix_b, &blocks_b, &block_manager)
            .unwrap();

        // Match prefix B fully
        let (matched_b, len_b) = radix.match_prefix(&prefix_b);
        assert_eq!(len_b, 4);
        assert_eq!(matched_b.len(), 2);

        // Match original prefix A still intact
        let (matched_a, len_a) = radix.match_prefix(&prefix_a);
        assert_eq!(len_a, 4);
        assert_eq!(matched_a.len(), 2);

        // Match shorter common prefix [101, 102]
        let (matched_common, len_common) = radix.match_prefix(&[101, 102]);
        assert_eq!(len_common, 2);
        assert_eq!(matched_common.len(), 1);
    }

    #[test]
    fn test_radix_tree_pinned_node_protection() {
        let block_manager = PagedBlockManager::new(64, 16);
        let radix = RadixTree::new(1000);

        let prefix = vec![1, 2, 3, 4];
        let blocks = block_manager.allocate_blocks(2).unwrap();
        radix
            .insert_prefix(&prefix, &blocks, &block_manager)
            .unwrap();

        // Pin block
        block_manager.pin_block(blocks[0]).unwrap();
        assert!(block_manager.is_block_pinned(blocks[0]));
        let evicted = radix.evict_lru(&block_manager, 64);
        assert_eq!(evicted, 0);

        // Unpin block
        block_manager.unpin_block(blocks[0]).unwrap();

        // Now eviction succeeds
        let evicted2 = radix.evict_lru(&block_manager, 64);
        assert_eq!(evicted2, 2);
    }
}
