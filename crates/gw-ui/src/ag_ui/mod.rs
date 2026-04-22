//! AG-UI transport: SSE outbound + HTTP POST inbound.
//!
//! See `docs/design-gw-ui.md` §8. This module is the wire-level contract
//! between gw and an AG-UI-speaking frontend.

pub mod adapter;
pub mod codec;
pub mod events;

pub use adapter::AgUiAdapter;
pub use codec::{loop_event_to_ag_ui, notification_to_ag_ui};
pub use events::{AgUiEvent, PostMessageBody};
