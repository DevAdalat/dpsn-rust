use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use rand::Rng;

use crate::data::tokenizer::CharTokenizer;
use crate::model::dpsn::DPSN;

pub struct TextGenerator<'a, B: Backend> {
    model: &'a DPSN<B>,
    tokenizer: &'a CharTokenizer,
    device: B::Device,
}

impl<'a, B: Backend> TextGenerator<'a, B> {
    pub fn new(model: &'a DPSN<B>, tokenizer: &'a CharTokenizer, device: B::Device) -> Self {
        TextGenerator {
            model,
            tokenizer,
            device,
        }
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f64) -> String {
        let mut tokens = self.tokenizer.encode(prompt);
        let context_length = self.model.context_length;

        for _ in 0..max_tokens {
            let start_idx = if tokens.len() > context_length {
                tokens.len() - context_length
            } else {
                0
            };

            let context: Vec<i64> = tokens[start_idx..].iter().map(|&x| x as i64).collect();

            let context_tensor = Tensor::<B, 1, Int>::from_ints(context.as_slice(), &self.device)
                .reshape([1, context.len()]);

            let logits = self.model.forward_inference(context_tensor);

            let last_logits = logits
                .slice([
                    0..1,
                    (context.len() - 1)..context.len(),
                    0..self.tokenizer.vocab_size,
                ])
                .reshape([self.tokenizer.vocab_size]);

            let scaled_logits = last_logits / temperature;
            let probs = activation::softmax(scaled_logits, 0);

            let next_token = self.sample_from_probs(probs);
            tokens.push(next_token);
        }

        self.tokenizer.decode(&tokens)
    }

    fn sample_from_probs(&self, probs: Tensor<B, 1>) -> usize {
        let probs_vec: Vec<f32> = probs.into_data().to_vec().unwrap();

        let mut rng = rand::thread_rng();
        let random_val: f32 = rng.gen();

        let mut cumsum = 0.0;
        for (idx, &prob) in probs_vec.iter().enumerate() {
            cumsum += prob;
            if random_val < cumsum {
                return idx;
            }
        }

        probs_vec.len() - 1
    }

    pub fn generate_greedy(&self, prompt: &str, max_tokens: usize) -> String {
        let mut tokens = self.tokenizer.encode(prompt);
        let context_length = self.model.context_length;

        for _ in 0..max_tokens {
            let start_idx = if tokens.len() > context_length {
                tokens.len() - context_length
            } else {
                0
            };

            let context: Vec<i64> = tokens[start_idx..].iter().map(|&x| x as i64).collect();

            let context_tensor = Tensor::<B, 1, Int>::from_ints(context.as_slice(), &self.device)
                .reshape([1, context.len()]);

            let logits = self.model.forward_inference(context_tensor);

            let last_logits = logits
                .slice([
                    0..1,
                    (context.len() - 1)..context.len(),
                    0..self.tokenizer.vocab_size,
                ])
                .reshape([self.tokenizer.vocab_size]);

            let argmax_data: Vec<i64> = last_logits.argmax(0).into_data().to_vec().unwrap();
            let next_token = argmax_data.first().copied().unwrap_or(0) as usize;
            tokens.push(next_token);
        }

        self.tokenizer.decode(&tokens)
    }
}
