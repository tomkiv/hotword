# Product Guidelines - Hotword Detection in Go

## Documentation & Code Style
- **Clarity over Complexity:** Documentation and code comments must prioritize clear, simple explanations of audio processing and mathematical logic, making the project accessible to hobbyists and non-experts.
- **Performance Rationale:** When implementing optimizations or specific memory management strategies for resource-constrained devices, document the reasoning and trade-offs.
- **Idiomatic Go:** Follow standard Go conventions (Gofmt, explicit error handling, etc.).

## CLI & User Experience
- **Minimalist & Functional:** CLI output should be clean and noise-free, suitable for both human readability and script integration.
- **Informative Progress:** Provide clear feedback during long-running tasks like model training or dataset processing (e.g., using progress bars).
- **Interactive Audio Feedback:** Include simple text-based visualizers (like VU meters) to assist users in verifying audio input and sensitivity.

## Architecture & Organization
- **Modular Library Design:** The core logic should be organized into reusable packages (e.g., `pkg/audio`, `pkg/model`, `pkg/train`) to facilitate integration into other Go applications.
- **Fail-Fast & Explicit Errors:** Use explicit error returns with actionable messages. Ensure the application fails fast if critical resources like audio hardware are unavailable.

## Dependency Management
- **Strict Minimalism:** Avoid external dependencies. Implement core functionality (audio processing, math, neural network layers) using Go's standard library or native implementations to ensure maximum portability and zero-config deployment.
