use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct AuthSession {
    id: u64,
    cancel_flag: Arc<AtomicBool>,
}

impl AuthSession {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::Relaxed)
    }
}

#[derive(Default)]
pub struct AuthControl {
    next_id: AtomicU64,
    current_id: AtomicU64,
    sessions: Mutex<HashMap<u64, Arc<AtomicBool>>>,
}

impl AuthControl {
    pub fn register(&self) -> AuthSession {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let cancel_flag = Arc::new(AtomicBool::new(false));

        self.sessions
            .lock()
            .expect("auth control mutex poisoned")
            .insert(id, cancel_flag.clone());
        self.current_id.store(id, Ordering::Relaxed);

        AuthSession { id, cancel_flag }
    }

    pub fn finish(&self, id: u64) {
        self.sessions
            .lock()
            .expect("auth control mutex poisoned")
            .remove(&id);

        let current_id = self.current_id.load(Ordering::Relaxed);
        if current_id == id {
            self.current_id.store(0, Ordering::Relaxed);
        }
    }

    pub fn cancel_current(&self) -> bool {
        let current_id = self.current_id.load(Ordering::Relaxed);
        if current_id == 0 {
            return false;
        }

        let sessions = self.sessions.lock().expect("auth control mutex poisoned");
        let Some(cancel_flag) = sessions.get(&current_id) else {
            return false;
        };

        cancel_flag.store(true, Ordering::Relaxed);
        true
    }
}
