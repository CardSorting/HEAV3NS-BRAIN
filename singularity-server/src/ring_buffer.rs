use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamToken {
    pub seq_no: u64,
    pub token: u32,
    pub text: String,
    pub is_finish: bool,
}

pub struct StreamRingBuffer {
    capacity: usize,
    buffer: RwLock<VecDeque<StreamToken>>,
    stream_buffers: RwLock<HashMap<u64, VecDeque<StreamToken>>>,
}

impl StreamRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: RwLock::new(VecDeque::with_capacity(capacity)),
            stream_buffers: RwLock::new(HashMap::new()),
        }
    }

    pub fn push(&self, token: StreamToken) {
        let mut buf = self.buffer.write();
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(token);
    }

    /// Replays tokens following the specified last_seq_no for network reconnect.
    pub fn replay_since(&self, last_seq_no: u64) -> Vec<StreamToken> {
        let buf = self.buffer.read();
        buf.iter()
            .filter(|t| t.seq_no > last_seq_no)
            .cloned()
            .collect()
    }

    /// Pushes a token to a specific stream buffer identified by stream_id.
    pub fn push_for_stream(&self, stream_id: u64, token: StreamToken) {
        let mut streams = self.stream_buffers.write();
        let buf = streams
            .entry(stream_id)
            .or_insert_with(|| VecDeque::with_capacity(self.capacity));
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(token.clone());
        drop(streams);
        // Also mirror to global buffer for backward compatibility
        self.push(token);
    }

    /// Replays tokens for a specific stream following the specified last_seq_no.
    pub fn replay_since_for_stream(&self, stream_id: u64, last_seq_no: u64) -> Vec<StreamToken> {
        let streams = self.stream_buffers.read();
        if let Some(buf) = streams.get(&stream_id) {
            buf.iter()
                .filter(|t| t.seq_no > last_seq_no)
                .cloned()
                .collect()
        } else {
            self.replay_since(last_seq_no)
        }
    }

    pub fn remove_stream(&self, stream_id: u64) {
        self.stream_buffers.write().remove(&stream_id);
    }

    pub fn all_tokens(&self) -> Vec<StreamToken> {
        self.buffer.read().iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_ring_buffer_isolation() {
        let ring = StreamRingBuffer::new(8);

        // Stream 1
        ring.push_for_stream(
            1,
            StreamToken {
                seq_no: 1,
                token: 10,
                text: "a".to_string(),
                is_finish: false,
            },
        );
        ring.push_for_stream(
            1,
            StreamToken {
                seq_no: 2,
                token: 11,
                text: "b".to_string(),
                is_finish: false,
            },
        );

        // Stream 2
        ring.push_for_stream(
            2,
            StreamToken {
                seq_no: 1,
                token: 20,
                text: "x".to_string(),
                is_finish: false,
            },
        );

        // Replay stream 1 since seq 1 should only return stream 1's token 2
        let replayed1 = ring.replay_since_for_stream(1, 1);
        assert_eq!(replayed1.len(), 1);
        assert_eq!(replayed1[0].text, "b");
        assert_eq!(replayed1[0].token, 11);

        // Replay stream 2 since seq 0 should only return stream 2's token 1
        let replayed2 = ring.replay_since_for_stream(2, 0);
        assert_eq!(replayed2.len(), 1);
        assert_eq!(replayed2[0].text, "x");
    }
}
