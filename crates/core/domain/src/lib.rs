//! Domain layer for AI concepts, commands, entities, errors, and utilities.

pub mod commands {
    /// Marker command for bootstrapping the domain layer.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Initialize;
}

pub mod entities {
    /// Minimal model identity for the first workspace slice.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ModelId(pub String);
}

pub mod errors {
    pub use kpodjito_core_error::Error;
}

pub mod utils {
    /// Placeholder domain utility namespace.
    pub fn identity<T>(value: T) -> T {
        value
    }
}