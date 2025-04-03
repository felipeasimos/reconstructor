mod app;

use app::ApplicationState;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event_loop::EventLoop,
    keyboard::{KeyCode},
};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let mut app = ApplicationState::new();
    let _ = event_loop.run_app(&mut app);
}
