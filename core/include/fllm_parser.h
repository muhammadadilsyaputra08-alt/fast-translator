#pragma once
#include "fllm_format.h"
#include <optional>
#include <string>

namespace fllm {

// Writes an in-memory FllmModel to disk in the .fllm binary layout.
// Returns true on success.
bool write_fllm(const std::string& path, const FllmModel& model);

// Reads and validates a .fllm file from disk, including checksum verification.
// Returns std::nullopt if the file is missing, malformed, or checksum fails.
std::optional<FllmModel> read_fllm(const std::string& path);

// Validates only the header (magic + version) without loading full sections.
// Useful for a fast pre-flight check before committing to a full load.
bool validate_header(const std::string& path);

} // namespace fllm
