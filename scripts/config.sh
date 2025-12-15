#!/bin/bash
#
# Scripts Configuration
#
# This file controls output behavior for all batch processing scripts.
# Settings can be overridden via environment variables.
#

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Whether to save batch results to files (true/false)
# Override with: BATCH_SAVE_RESULTS=false ./scripts/monitor_batch.sh
BATCH_SAVE_RESULTS="${BATCH_SAVE_RESULTS:-true}"

# Whether to display batch results in terminal (true/false)
# Override with: BATCH_DISPLAY_RESULTS=false ./scripts/monitor_batch.sh
BATCH_DISPLAY_RESULTS="${BATCH_DISPLAY_RESULTS:-true}"

# Directory for saving batch outputs (results, errors, analysis)
# Override with: BATCH_OUTPUT_DIR=/custom/path ./scripts/monitor_batch.sh
BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-./data/batch_outputs}"

# Directory for checkpoint files
# Override with: CHECKPOINT_DIR=/custom/checkpoint/path
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/app/data/checkpoints}"

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

# Polling interval for monitoring (seconds)
BATCH_POLL_INTERVAL="${BATCH_POLL_INTERVAL:-30}"

# Whether to show detailed progress information
BATCH_SHOW_DETAILED_PROGRESS="${BATCH_SHOW_DETAILED_PROGRESS:-true}"

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# Whether to save analysis reports (true/false)
BATCH_SAVE_ANALYSIS="${BATCH_SAVE_ANALYSIS:-true}"

# Whether to display analysis reports in terminal (true/false)
BATCH_DISPLAY_ANALYSIS="${BATCH_DISPLAY_ANALYSIS:-true}"

# Analysis output subdirectory name format (strftime compatible)
BATCH_ANALYSIS_SUBDIR_FORMAT="${BATCH_ANALYSIS_SUBDIR_FORMAT:-batch_analysis_%Y%m%d_%H%M%S}"

# =============================================================================
# FILE NAMING CONFIGURATION
# =============================================================================

# Result file naming pattern (uses job_id)
BATCH_RESULTS_FILE_PATTERN="${BATCH_RESULTS_FILE_PATTERN:-batch_results_%s.json}"

# Error file naming pattern (uses job_id)
BATCH_ERROR_FILE_PATTERN="${BATCH_ERROR_FILE_PATTERN:-batch_error_%s.json}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Ensure output directory exists
ensure_output_dir() {
    if [ "$BATCH_SAVE_RESULTS" = "true" ] || [ "$BATCH_SAVE_ANALYSIS" = "true" ]; then
        mkdir -p "$BATCH_OUTPUT_DIR"
    fi
}

# Get full path for results file
get_results_file_path() {
    local job_id="$1"
    if [ "$BATCH_SAVE_RESULTS" = "true" ]; then
        ensure_output_dir
        printf "${BATCH_OUTPUT_DIR}/${BATCH_RESULTS_FILE_PATTERN}" "$job_id"
    else
        echo ""
    fi
}

# Get full path for error file
get_error_file_path() {
    local job_id="$1"
    if [ "$BATCH_SAVE_RESULTS" = "true" ]; then
        ensure_output_dir
        printf "${BATCH_OUTPUT_DIR}/${BATCH_ERROR_FILE_PATTERN}" "$job_id"
    else
        echo ""
    fi
}

# Get full path for analysis directory
get_analysis_dir_path() {
    if [ "$BATCH_SAVE_ANALYSIS" = "true" ]; then
        ensure_output_dir
        echo "${BATCH_OUTPUT_DIR}/$(date +"${BATCH_ANALYSIS_SUBDIR_FORMAT}")"
    else
        # Use temp directory for display-only mode
        mktemp -d
    fi
}

# Save JSON to file if saving enabled
save_json() {
    local json_content="$1"
    local file_path="$2"
    local description="$3"

    if [ "$BATCH_SAVE_RESULTS" = "true" ] && [ -n "$file_path" ]; then
        echo "$json_content" | jq '.' > "$file_path"
        echo -e "${BLUE}${description} saved to: ${GREEN}${file_path}${NC}"
    fi
}

# Display JSON if display enabled
display_json() {
    local json_content="$1"
    local title="$2"
    local jq_filter="${3:-.}"

    if [ "$BATCH_DISPLAY_RESULTS" = "true" ]; then
        if [ -n "$title" ]; then
            echo ""
            echo -e "${BLUE}${title}:${NC}"
        fi
        echo "$json_content" | jq "$jq_filter"
    fi
}

# Save and/or display JSON based on configuration
save_and_display_json() {
    local json_content="$1"
    local file_path="$2"
    local description="$3"
    local title="$4"
    local jq_filter="${5:-.}"

    save_json "$json_content" "$file_path" "$description"
    display_json "$json_content" "$title" "$jq_filter"
}

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# Example 1: Display only, don't save
#   BATCH_SAVE_RESULTS=false ./scripts/monitor_batch.sh
#
# Example 2: Save only, don't display
#   BATCH_DISPLAY_RESULTS=false ./scripts/monitor_batch.sh
#
# Example 3: Custom output directory
#   BATCH_OUTPUT_DIR=/tmp/my_outputs ./scripts/monitor_batch.sh
#
# Example 4: Combination
#   BATCH_OUTPUT_DIR=/custom/path BATCH_SAVE_RESULTS=true ./scripts/monitor_batch.sh
#
# Example 5: Quick monitoring (no save, display only)
#   BATCH_SAVE_RESULTS=false BATCH_SAVE_ANALYSIS=false ./scripts/monitor_batch.sh
#
# =============================================================================
