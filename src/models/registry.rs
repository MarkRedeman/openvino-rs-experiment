use anyhow::{Result, bail};
use std::collections::HashMap;

use crate::models::model_wrapper::ModelWrapper;

pub struct ModelRegistry {
    models: HashMap<String, ModelWrapper>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn load(&mut self, id: impl Into<String>, model: ModelWrapper) -> Result<()> {
        let key = id.into();
        if self.models.contains_key(&key) {
            bail!("model id already loaded: {key}");
        }
        self.models.insert(key, model);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&ModelWrapper> {
        self.models.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut ModelWrapper> {
        self.models.get_mut(id)
    }

    pub fn unload(&mut self, id: &str) -> Option<ModelWrapper> {
        self.models.remove(id)
    }

    pub fn list(&self) -> Vec<&str> {
        self.models.keys().map(|k| k.as_str()).collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
