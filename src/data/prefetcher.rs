use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;

use super::dataset::{CharDataset, DataLoader};

pub struct DataPrefetcher {
    receiver: Receiver<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
}

impl DataPrefetcher {
    pub fn new(dataset: CharDataset, batch_size: usize, shuffle: bool, buffer_size: usize) -> Self {
        let (sender, receiver): (
            SyncSender<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
            Receiver<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
        ) = sync_channel(buffer_size);

        thread::spawn(move || {
            let mut dataloader = DataLoader::new(&dataset, batch_size, shuffle);
            loop {
                let batch = dataloader.next_batch();
                if sender.send(batch).is_err() {
                    break;
                }
            }
        });

        DataPrefetcher { receiver }
    }

    pub fn next(&self) -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        self.receiver.recv().ok()
    }
}
